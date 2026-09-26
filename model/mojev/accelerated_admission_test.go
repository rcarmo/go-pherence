package mojev

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// Explicit, separately selected stress gate. One checkpoint and one accelerator
// per process; ordinary tests never allocate released-size weights or scratch.
func TestMoJevAcceleratedAdmission(t *testing.T) {
	backend := os.Getenv("GO_PHERENCE_MOJEV_ADMISSION_BACKEND")
	if backend == "" {
		t.Skip("set GO_PHERENCE_MOJEV_ADMISSION_BACKEND=simd or nvidia and GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	}
	if backend != "simd" && backend != "nvidia" {
		t.Fatal("unsupported admission backend")
	}
	// Instrumented runs can select one round; longer opt-in retention runs
	// reuse the same bounded requests, ownership and cancellation checks.
	// Keep the ordinary released/race gate at three rounds and report overrides.
	rounds := 3
	if value := os.Getenv("GO_PHERENCE_MOJEV_ADMISSION_ROUNDS"); value != "" {
		var err error
		rounds, err = strconv.Atoi(value)
		if err != nil || rounds < 1 || rounds > 60 {
			t.Fatal("admission rounds must be 1..60")
		}
	}
	t.Logf("ADMISSION_CONFIG backend=%s capacity=512 retention_rounds=%d callers=8", backend, rounds)
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint directory required")
	}
	fixture := nativeFixture(t)
	for name, want := range map[string]string{"model.safetensors": fixture.WeightSHA, "config.json": fixture.ConfigSHA} {
		file, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, file)
		file.Close()
		if err != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("checkpoint hash", name, err)
		}
	}
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	cpu, err := LoadTextScorer(src, config)
	if err != nil {
		t.Fatal(err)
	}
	src.Close()
	type executor interface {
		ScoreEncoded(EncodedRow) ([][]float32, error)
		ScoreEncodedContext(context.Context, EncodedRow) ([][]float32, error)
	}
	var scorer executor
	var lock, unlock func()
	var resident int64
	var gpu *NVIDIATextScorer
	if backend == "simd" {
		s, e := NewSIMDTextScorer(cpu, 512)
		if e != nil {
			t.Fatal(e)
		}
		scorer = s
		lock, unlock = s.mu.Lock, s.mu.Unlock
	} else {
		gpu, err = NewNVIDIATextScorer(cpu, 512)
		if err != nil {
			if gpu != nil {
				_ = gpu.Close()
			}
			t.Fatal(err)
		}
		defer func() {
			if e := gpu.Close(); e != nil {
				t.Error(e)
			}
		}()
		scorer = gpu
		lock, unlock = gpu.mu.Lock, gpu.mu.Unlock
		resident = gpu.ResidentBytes()
	}
	type snapshot struct {
		Phase                                         string
		HeapAlloc, HeapInuse, HeapObjects, StackInuse uint64
		RSSKiB, HWMKiB                                uint64
		Goroutines                                    int
		DeviceResident                                int64
	}
	snapshots := []snapshot{}
	measure := func(phase string) snapshot {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		s := snapshot{Phase: phase, HeapAlloc: m.HeapAlloc, HeapInuse: m.HeapInuse, HeapObjects: m.HeapObjects, StackInuse: m.StackInuse, Goroutines: runtime.NumGoroutine(), DeviceResident: resident}
		if data, e := os.ReadFile("/proc/self/status"); e == nil {
			for _, line := range strings.Split(string(data), "\n") {
				parts := strings.Fields(line)
				if len(parts) < 2 {
					continue
				}
				v, _ := strconv.ParseUint(parts[1], 10, 64)
				if parts[0] == "VmRSS:" {
					s.RSSKiB = v
				}
				if parts[0] == "VmHWM:" {
					s.HWMKiB = v
				}
			}
		}
		snapshots = append(snapshots, s)
		b, _ := json.Marshal(s)
		t.Logf("ADMISSION %s", b)
		runtime.KeepAlive(scorer)
		runtime.KeepAlive(cpu)
		return s
	}
	measure("loaded")
	ids := func(n, seed int) []int {
		x := make([]int, n)
		for i := range x {
			x[i] = 100 + (i*7+seed)%800
		}
		return x
	}
	maximum := EncodedRow{State: ids(128, 0), Questions: [][]int{ids(128, 1)}, Candidates: [][][]int{{ids(256, 2), ids(256, 3)}}}
	packed := EncodedRow{State: ids(64, 4), Questions: [][]int{ids(64, 5)}, Candidates: make([][][]int, 1)}
	packed.Candidates[0] = make([][]int, 64)
	for i := range packed.Candidates[0] {
		packed.Candidates[0][i] = ids(62, 6+i)
	} // 128+64*62=4096
	inputs := []EncodedRow{fixture.Cases["base"].Row, maximum, packed}
	wants := make([][][]float32, len(inputs))
	score := func(row EncodedRow) [][]float32 {
		t.Helper()
		started := time.Now()
		got, e := scorer.ScoreEncoded(row)
		if e != nil {
			t.Fatal(e)
		}
		if len(got) != len(row.Questions) {
			t.Fatal("output geometry")
		}
		for f := range got {
			if len(got[f]) != len(row.Candidates[f]) {
				t.Fatal("candidate geometry")
			}
			for _, v := range got[f] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite output")
				}
			}
		}
		t.Logf("SCORE fields=%d candidates=%d elapsed_ms=%.3f", len(row.Questions), len(row.Candidates[0]), float64(time.Since(started))/1e6)
		return got
	}
	for i, r := range inputs {
		wants[i] = score(r)
	}
	// The new sizes are stress/determinism cases, not new independent oracles.
	for f := range wants[0] {
		for i, v := range wants[0][f] {
			if math.Abs(float64(v-fixture.Cases["base"].Logits[f][i])) > 3e-4 {
				t.Fatal("released base parity")
			}
		}
	}
	baseline := measure("warm_512_and_4096")
	var held [][][]float32
	var copies [][][]float32
	for round := 0; round < rounds; round++ {
		// Reject over-capacity branches and over-total requests without partial output.
		bad := maximum
		bad.State = ids(129, 0)
		if out, e := scorer.ScoreEncoded(bad); e == nil || out != nil || !strings.Contains(e.Error(), "capacity") {
			t.Fatal("wrong 513-token rejection", e)
		}
		bad = packed
		bad.State = ids(65, 4)
		if out, e := scorer.ScoreEncoded(bad); e == nil || out != nil || !strings.Contains(e.Error(), "branch-local text length") {
			t.Fatal("wrong 4097-token rejection", e)
		}
		for _, index := range []int{1, 2, 0} {
			got := score(inputs[index])
			if !reflect.DeepEqual(got, wants[index]) {
				t.Fatal("size/error recovery mismatch", round, index)
			}
			clone := make([][]float32, len(got))
			for f := range got {
				clone[f] = append([]float32(nil), got[f]...)
			}
			held = append(held, got)
			copies = append(copies, clone)
		}
		if !reflect.DeepEqual(held, copies) {
			t.Fatal("retained output mutation")
		}
		s := measure(fmt.Sprintf("retained_round_%d", round+1))
		// A bounded regression ceiling; only ~300 logits are intentionally retained.
		if s.HeapAlloc > baseline.HeapAlloc+32<<20 {
			t.Fatal("post-GC heap grew beyond32MiB allowance", s.HeapAlloc, baseline.HeapAlloc)
		}
		if gpu != nil && gpu.ResidentBytes() != resident {
			t.Fatal("device allocation grew")
		}
	}
	var wg sync.WaitGroup
	startTogether := make(chan struct{})
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-startTogether
			which := index % 2
			got, e := scorer.ScoreEncoded(inputs[which])
			if e != nil || !reflect.DeepEqual(got, wants[which]) {
				t.Error("concurrent mismatch", index, e)
			}
		}(worker)
	}
	close(startTogether)
	wg.Wait()
	// Eight canceled callers are not allowed to wait for the scratch owner.
	lock()
	done := make(chan error, 8)
	for range 8 {
		go func() {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			out, e := scorer.ScoreEncodedContext(ctx, maximum)
			if out != nil {
				e = errors.New("partial canceled scores")
			}
			done <- e
		}()
	}
	for range 8 {
		select {
		case e := <-done:
			if !errors.Is(e, context.Canceled) {
				t.Error(e)
			}
		case <-time.After(5 * time.Second):
			unlock()
			t.Fatal("cancelled waiter blocked")
		}
	}
	unlock()
	testReleasedCancellation(t, scorer)
	largeCtx, cancelLarge := context.WithCancel(context.Background())
	probe := &pollCancelContext{Context: largeCtx, cancel: cancelLarge}
	probe.remaining.Store(10)
	out, cancelErr := scorer.ScoreEncodedContext(probe, maximum)
	cancelLarge()
	if !errors.Is(cancelErr, context.Canceled) || out != nil {
		t.Fatal("512-token active cancellation", cancelErr)
	}
	if !reflect.DeepEqual(score(maximum), wants[1]) {
		t.Fatal("large request after cancel")
	}
	final := measure("concurrent_and_cancel_recovered")
	if final.HeapAlloc > baseline.HeapAlloc+32<<20 {
		t.Fatal("retained memory regression after concurrency")
	}
	if final.Goroutines > baseline.Goroutines+2 {
		t.Fatal("worker goroutines retained", final.Goroutines, baseline.Goroutines)
	}
	runtime.KeepAlive(held)
	runtime.KeepAlive(copies)
	held, copies = nil, nil
	measure("outputs_released")
	runtime.KeepAlive(wants)
	runtime.KeepAlive(scorer)
	runtime.KeepAlive(cpu)
	if gpu != nil {
		// Close is documented as idempotent; the deferred second call is also
		// checked. Keep it registered so earlier failures still release buffers.
		before, _ := nvidia.MemInfo()
		if err := gpu.Close(); err != nil {
			t.Fatal(err)
		}
		after, _ := nvidia.MemInfo()
		if gpu.ResidentBytes() != 0 {
			t.Fatal("closed residency")
		}
		t.Logf("DEVICE free_before=%d free_after=%d expected_released=%d", before, after, resident)
		resident = 0
		measure("gpu_closed")
	}
	if p := os.Getenv("GO_PHERENCE_MOJEV_ADMISSION_REPORT"); p != "" {
		data, e := json.MarshalIndent(struct {
			Backend         string
			RetentionRounds int
			Snapshots       []snapshot
			Logits          [][][]float32
		}{backend, rounds, snapshots, wants}, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(p, data, 0600); e != nil {
			t.Fatal(e)
		}
	}
}
