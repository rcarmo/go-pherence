package mojev

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/weights"
)

func cpuContextFixture(t *testing.T) longTextFixture {
	t.Helper()
	for path, want := range map[string]string{"../../scripts/mojev_oracle_cpu_context.py": "e4f4ba0ff7240806dc023bad215f70167b26134d2a9599ffcc0e588c1b4607f5", "testdata/native_cpu_context.json": "495d452b956ca618fbdd35424fcbba0a29cfc047db9c923ccb1c5412163382e8"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != want {
			t.Fatal("fixture hash", path)
		}
	}
	data, err := os.ReadFile("testdata/native_cpu_context.json")
	if err != nil {
		t.Fatal(err)
	}
	var f longTextFixture
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Revision != SourceRevision || f.Policy != "f32-branch-local-positions-full-tree-causal-linear" || len(f.Cases) != 5 {
		t.Fatal("fixture provenance")
	}
	return f
}

func TestCPUContextFixture(t *testing.T) {
	f := cpuContextFixture(t)
	for name, wantPath := range map[string]int{"path_513": 513, "path_1024": 1024, "max_request": 4095, "shared_tree": 3072, "encoder_4096": 4096} {
		c, ok := f.Cases[name]
		if !ok || len(c.Row.Questions) != 1 || len(c.Row.Candidates) != 1 || len(c.Logits) != 1 {
			t.Fatal("shape", name)
		}
		count := 2
		if name == "encoder_4096" {
			count = 1
		}
		if len(c.Row.Candidates[0]) != count || len(c.Logits[0]) != count || len(c.Hidden) != count || len(c.Positions) != count {
			t.Fatal("candidate shape", name)
		}
		ns, nq := len(c.Row.State), len(c.Row.Questions[0])
		maxPath, total := 0, ns+nq
		for i, ids := range c.Row.Candidates[0] {
			n := ns + nq + len(ids)
			maxPath = max(maxPath, n)
			total += len(ids)
			if !reflect.DeepEqual(c.Positions[i], []int{ns - 1, ns + nq - 1, n - 1}) || len(c.Hidden[i]) != 3 {
				t.Fatal("positions", name)
			}
			if math.IsNaN(float64(c.Logits[0][i])) || math.IsInf(float64(c.Logits[0][i]), 0) {
				t.Fatal("nonfinite logit")
			}
			for _, row := range c.Hidden[i] {
				if len(row) != 1024 {
					t.Fatal("hidden width")
				}
				for _, v := range row {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite hidden")
					}
				}
			}
		}
		if maxPath != wantPath || total > 4096 {
			t.Fatal("context geometry", name, maxPath, total)
		}
	}
}

// Opt-in: one approved checkpoint process. This is CPU-only; GPU capacity stays512.
func TestReleasedCPUContext(t *testing.T) {
	if os.Getenv("GO_PHERENCE_MOJEV_CPU_CONTEXT") != "1" {
		t.Skip("set GO_PHERENCE_MOJEV_CPU_CONTEXT=1 and GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	}
	f := cpuContextFixture(t)
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint required")
	}
	for name, want := range map[string]string{"config.json": f.ConfigSHA, "model.safetensors": f.WeightSHA} {
		file, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, file)
		file.Close()
		if err != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("asset hash", name, err)
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
	s, err := NewSIMDTextScorer(cpu, 4096)
	if err != nil {
		t.Fatal(err)
	}
	// Drop raw projection weights; the prepared scorer retains the host readout.
	cpu = nil
	runtime.GC()
	snapshot := func(label string) {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		t.Logf("MEMORY %s heap=%d inuse=%d goroutines=%d", label, m.HeapAlloc, m.HeapInuse, runtime.NumGoroutine())
	}
	snapshot("loaded")
	var maxLogit, maxHidden float64
	results := make(map[string][][]float32)
	for _, name := range []string{"path_513", "path_1024", "max_request", "shared_tree", "encoder_4096"} {
		c := f.Cases[name]
		started := time.Now()
		ns, nq := len(c.Row.State), len(c.Row.Questions[0])
		// Direct path probe reaches4096 even though public scoring requires >=2 options
		// inside its unchanged4096-total-token budget.
		for i, candidate := range c.Row.Candidates[0] {
			ids := append(append(append([]int{}, c.Row.State...), c.Row.Questions[0]...), candidate...)
			hidden, e := s.encodeBranch(TextBranch{IDs: ids, StateLen: ns, QuestionLen: nq})
			if e != nil {
				t.Fatal(name, e)
			}
			for sample, pos := range c.Positions[i] {
				for j, v := range c.Hidden[i][sample] {
					maxHidden = math.Max(maxHidden, math.Abs(float64(hidden[pos*1024+j]-v)))
				}
			}
		}
		if name != "encoder_4096" {
			got, e := s.ScoreEncoded(c.Row)
			if e != nil {
				t.Fatal(name, e)
			}
			results[name] = got
			if len(got) != 1 || len(got[0]) != 2 {
				t.Fatal("output geometry")
			}
			for i, v := range got[0] {
				maxLogit = math.Max(maxLogit, math.Abs(float64(v-c.Logits[0][i])))
			}
		}
		if math.IsNaN(maxLogit) || math.IsNaN(maxHidden) || maxLogit > 3e-4 || maxHidden > 2e-3 {
			t.Fatal("reference parity", name, maxLogit, maxHidden)
		}
		t.Logf("CONTEXT %s max_logits=%g max_hidden=%g elapsed_s=%.3f", name, maxLogit, maxHidden, time.Since(started).Seconds())
	}
	// Confirm final-normalized hidden rows for the actual4096-token grouped tree.
	c := f.Cases["shared_tree"]
	prefix := len(c.Row.State) + len(c.Row.Questions[0])
	ids := append(append(append(append([]int{}, c.Row.State...), c.Row.Questions[0]...), c.Row.Candidates[0][0]...), c.Row.Candidates[0][1]...)
	for i, id := range ids {
		s.rows[i] = s.cpu.embedding[id*1024 : (id+1)*1024]
	}
	hidden := s.hidden[:len(ids)*1024]
	if e := s.branch.ForwardTreeInto(hidden, s.rows[:len(ids)], len(c.Row.State), len(c.Row.Questions[0]), []int{prefix + len(c.Row.Candidates[0][0]), len(ids)}, s.cpu.rope, s.cpu.eps); e != nil {
		t.Fatal(e)
	}
	s.cpu.normaliseFinal(hidden)
	for i := range c.Hidden {
		for sample, pos := range c.Positions[i] {
			if pos >= prefix && i == 1 {
				pos += len(c.Row.Candidates[0][0])
			}
			for j, v := range c.Hidden[i][sample] {
				maxHidden = math.Max(maxHidden, math.Abs(float64(hidden[pos*1024+j]-v)))
			}
		}
	}
	if maxHidden > 2e-3 || math.IsNaN(maxHidden) {
		t.Fatal("grouped hidden", maxHidden)
	}
	held := results["shared_tree"]
	saved := [][]float32{append([]float32(nil), held[0]...)}
	changed := c.Row
	changed.Candidates = [][][]int{{append([]int(nil), c.Row.Candidates[0][0]...), append([]int(nil), c.Row.Candidates[0][1]...)}}
	changed.Candidates[0][1][0] = 1234
	got, e := s.ScoreEncoded(changed)
	if e != nil || got[0][0] != held[0][0] {
		t.Fatal("sibling isolation", e)
	}
	reordered := c.Row
	reordered.Candidates = [][][]int{{c.Row.Candidates[0][1], c.Row.Candidates[0][0]}}
	got, e = s.ScoreEncoded(reordered)
	if e != nil || !reflect.DeepEqual(got[0], []float32{held[0][1], held[0][0]}) {
		t.Fatal("permutation", e)
	}
	// Over-limit errors must not publish output. Then cancel in-flight work and
	// prove deterministic recovery, including the reused metadata buffers.
	bad := f.Cases["max_request"].Row
	bad.State = append(append([]int(nil), bad.State...), 100)
	if out, e := s.ScoreEncoded(bad); e == nil || out != nil {
		t.Fatal("4097 total accepted")
	}
	if out, e := s.encodeBranch(TextBranch{IDs: make([]int, 4097), StateLen: 3072, QuestionLen: 1024}); e == nil || out != nil {
		t.Fatal("4097 path accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	probe := &pollCancelContext{Context: ctx, cancel: cancel}
	probe.remaining.Store(10)
	out, e := s.ScoreEncodedContext(probe, c.Row)
	cancel()
	if e != context.Canceled || out != nil {
		t.Fatal("active cancellation", e)
	}
	got, e = s.ScoreEncoded(c.Row)
	if e != nil || !reflect.DeepEqual(got, held) {
		t.Fatal("cancel recovery", e)
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, e := s.ScoreEncoded(c.Row)
			if e != nil || !reflect.DeepEqual(got, held) {
				t.Error("concurrency", e)
			}
		}()
	}
	wg.Wait()
	if !reflect.DeepEqual(held, saved) {
		t.Fatal("retained outputs mutated")
	}
	snapshot("reused_concurrent_cancel")
	t.Logf("CONTEXT_FINAL max_logits=%g max_hidden=%g", maxLogit, maxHidden)
	runtime.KeepAlive(s)
}
