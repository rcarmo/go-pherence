package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/weights"
)

// Each explicit run uses one accelerator and capacity, hence one checkpoint
// process. Reordering and substitution run concurrently on the same scorer.
func TestReleasedGroupedTextScorer(t *testing.T) {
	backend := os.Getenv("GO_PHERENCE_MOJEV_GROUPED_BACKEND")
	if backend == "" {
		t.Skip("set GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd|nvidia and checkpoint path; optional CAPACITY=256|512")
	}
	if backend != "simd" && backend != "nvidia" {
		t.Fatal("unsupported grouped backend")
	}
	capacity := 512
	if v := os.Getenv("GO_PHERENCE_MOJEV_GROUPED_CAPACITY"); v != "" {
		var err error
		capacity, err = strconv.Atoi(v)
		if err != nil || (capacity != 256 && capacity != 512) {
			t.Fatal("grouped capacity must be256 or512")
		}
	}
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint required")
	}
	f := groupedFixture(t)
	for name, want := range map[string]string{"config.json": f.ConfigSHA, "model.safetensors": f.WeightSHA} {
		file, e := os.Open(filepath.Join(dir, name))
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		_, e = io.Copy(h, file)
		file.Close()
		if e != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("asset hash", name, e)
		}
	}
	config, e := os.ReadFile(filepath.Join(dir, "config.json"))
	if e != nil {
		t.Fatal(e)
	}
	src, e := weights.OpenSafetensors(dir)
	if e != nil {
		t.Fatal(e)
	}
	defer src.Close()
	cpu, e := LoadTextScorer(src, config)
	if e != nil {
		t.Fatal(e)
	}
	src.Close()
	var score func(EncodedRow) ([][]float32, error)
	var encodeTree func(EncodedRow, int, int) ([]float32, error)
	if backend == "simd" {
		s, e := NewSIMDTextScorer(cpu, capacity)
		if e != nil {
			t.Fatal(e)
		}
		score = s.ScoreEncoded
		encodeTree = func(row EncodedRow, first, last int) ([]float32, error) {
			s.mu.Lock()
			defer s.mu.Unlock()
			n := 0
			add := func(ids []int) {
				for _, id := range ids {
					s.rows[n] = cpu.embedding[id*1024 : (id+1)*1024]
					n++
				}
			}
			add(row.State)
			add(row.Questions[0])
			ends := make([]int, last-first)
			for c := first; c < last; c++ {
				add(row.Candidates[0][c])
				ends[c-first] = n
			}
			out := s.hidden[:n*1024]
			if err := s.branch.ForwardTreeInto(out, s.rows[:n], 64, 64, ends, cpu.rope, cpu.eps); err != nil {
				return nil, err
			}
			cpu.normaliseFinal(out)
			return append([]float32(nil), out...), nil
		}
	} else {
		g, e := NewNVIDIATextScorer(cpu, capacity)
		if e != nil {
			if g != nil {
				_ = g.Close()
			}
			t.Fatal(e)
		}
		defer func() {
			if err := g.Close(); err != nil {
				t.Error(err)
			}
		}()
		score = g.ScoreEncoded
		encodeTree = func(row EncodedRow, first, last int) ([]float32, error) {
			g.mu.Lock()
			defer g.mu.Unlock()
			ids := append(append([]int{}, row.State...), row.Questions[0]...)
			ends := make([]int, last-first)
			for c := first; c < last; c++ {
				ids = append(ids, row.Candidates[0][c]...)
				ends[c-first] = len(ids)
			}
			out, err := g.encodeTree(TextBranch{IDs: ids, StateLen: 64, QuestionLen: 64}, ends)
			if err != nil {
				return nil, err
			}
			return append([]float32(nil), out...), nil
		}
	}
	maxError := func(got, want [][]float32) float64 {
		t.Helper()
		if len(got) != 1 || len(got[0]) != 64 {
			t.Fatal("result shape")
		}
		var d float64
		for c, v := range got[0] {
			d = math.Max(d, math.Abs(float64(v-want[0][c])))
		}
		return d
	}
	started := time.Now()
	got, err := score(f.Row)
	if err != nil {
		t.Fatal(err)
	}
	logitDiff := maxError(got, f.Logits)
	if math.IsNaN(logitDiff) || logitDiff > 3e-4 {
		t.Fatal("grouped oracle logits", logitDiff)
	}
	retained := append([]float32(nil), got[0]...)
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	// Each sample comes from an actual packed tree containing its neighbours,
	// not a separate candidate forward. Local branch position is remapped to it.
	var hiddenDiff float64
	for _, sample := range f.HiddenSamples {
		first := 0
		last := 0
		for first < len(f.Row.Candidates[0]) {
			last = candidateTreeEnd(f.Row.Candidates[0], first, 128, capacity)
			if sample.Candidate < last {
				break
			}
			first = last
		}
		hidden, e := encodeTree(f.Row, first, last)
		if e != nil {
			t.Fatal(e)
		}
		for i, p := range sample.Positions {
			if p >= 128 {
				p += (sample.Candidate - first) * 62
			}
			for j, v := range sample.Hidden[i] {
				hiddenDiff = math.Max(hiddenDiff, math.Abs(float64(hidden[p*1024+j]-v)))
			}
		}
	}
	if math.IsNaN(hiddenDiff) || hiddenDiff > 2e-3 {
		t.Fatal("grouped oracle hidden", hiddenDiff)
	}
	reverse := f.Row
	reverse.Candidates = [][][]int{append([][]int(nil), f.Row.Candidates[0]...)}
	for i, j := 0, 63; i < j; i, j = i+1, j-1 {
		reverse.Candidates[0][i], reverse.Candidates[0][j] = reverse.Candidates[0][j], reverse.Candidates[0][i]
	}
	changed := f.Row
	changed.Candidates = [][][]int{append([][]int(nil), f.Row.Candidates[0]...)}
	changed.Candidates[0][63] = append([]int(nil), changed.Candidates[0][63]...)
	changed.Candidates[0][63][0] = f.SiblingChanged.Token
	rows := []EncodedRow{reverse, changed}
	outputs := make([][][]float32, 2)
	errs := make([]error, 2)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range rows {
		wg.Add(1)
		go func(i int) { defer wg.Done(); <-start; outputs[i], errs[i] = score(rows[i]) }(i)
	}
	close(start)
	wg.Wait()
	for i, e := range errs {
		if e != nil {
			t.Fatal("concurrent maximum request", i, e)
		}
		if len(outputs[i]) != 1 || len(outputs[i][0]) != 64 {
			t.Fatal("concurrent shape")
		}
	}
	for c, v := range got[0] {
		if outputs[0][0][63-c] != v {
			t.Fatal("group-boundary reorder isolation", c)
		}
		if c != 63 && outputs[1][0][c] != v {
			t.Fatal("sibling contamination", c)
		}
	}
	changedDiff := math.Abs(float64(outputs[1][0][63] - f.SiblingChanged.Logit))
	if math.IsNaN(changedDiff) || changedDiff > 3e-4 {
		t.Fatal("changed-sibling oracle", changedDiff)
	}
	if !reflect.DeepEqual(got[0], retained) {
		t.Fatal("retained results mutated")
	}
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	if after.HeapAlloc > before.HeapAlloc+32<<20 {
		t.Fatal("bounded retention regression", before.HeapAlloc, after.HeapAlloc)
	}
	runtime.KeepAlive(score)
	runtime.KeepAlive(encodeTree)
	runtime.KeepAlive(cpu)
	runtime.KeepAlive(got)
	runtime.KeepAlive(outputs)
	t.Logf("GROUPED backend=%s capacity=%d logits_max=%g changed_max=%g hidden_max=%g heap_before=%d heap_after=%d elapsed_s=%.3f", backend, capacity, logitDiff, changedDiff, hiddenDiff, before.HeapAlloc, after.HeapAlloc, time.Since(started).Seconds())
	if path := os.Getenv("GO_PHERENCE_MOJEV_GROUPED_REPORT"); path != "" {
		report := struct {
			Backend                          string
			Capacity                         int
			Logits                           [][]float32
			MaxLogits, MaxChanged, MaxHidden float64
			HeapBefore, HeapAfter            uint64
		}{backend, capacity, got, logitDiff, changedDiff, hiddenDiff, before.HeapAlloc, after.HeapAlloc}
		data, e := json.MarshalIndent(report, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(path, data, 0600); e != nil {
			t.Fatal(e)
		}
	}
}
