package mojev

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"weak"

	"github.com/rcarmo/go-pherence/loader/weights"
	"github.com/rcarmo/go-pherence/model/qwen"
)

func mojevNVIDIALifetimeCheckpoint(t *testing.T) (string, nativeTextFixture, []byte) {
	t.Helper()
	if os.Getenv("GO_PHERENCE_MOJEV_NVIDIA") != "1" {
		t.Skip("set GO_PHERENCE_MOJEV_NVIDIA=1")
	}
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint required")
	}
	fixture := nativeFixture(t)
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(config)); got != fixture.ConfigSHA {
		t.Fatalf("config hash mismatch: got %s want %s", got, fixture.ConfigSHA)
	}
	file, err := os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	_, err = io.Copy(h, file)
	closeErr := file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != fixture.WeightSHA {
		t.Fatalf("weight hash mismatch: got %s want %s", got, fixture.WeightSHA)
	}
	return dir, fixture, config
}

//go:noinline
func newNVIDIATextScorerLifetimeSubject(t *testing.T, dir string, config []byte, base nativeTextCase) (*NVIDIATextScorer, weak.Pointer[qwen.Qwen35BaseModel]) {
	t.Helper()
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			_ = src.Close()
		}
	}()
	cpu, err := LoadTextScorer(src, config)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	if cpu == nil || cpu.model == nil || cpu.head == nil || len(cpu.embedding) == 0 {
		t.Fatal("invalid CPU scorer")
	}
	model := cpu.model
	w := weak.Make(model)
	g, err := NewNVIDIATextScorer(cpu, 256)
	if err != nil {
		if g != nil {
			_ = g.Close()
		}
		t.Fatal(err)
	}
	ok := false
	defer func() {
		if !ok && g != nil {
			_ = g.Close()
		}
	}()
	if g == nil || g.cpu == nil {
		t.Fatal("nil GPU scorer")
	}
	if g.cpu == cpu {
		t.Fatal("GPU scorer retained caller TextScorer")
	}
	if cpu.model != model {
		t.Fatal("constructor mutated caller model")
	}
	if g.cpu.model != nil {
		t.Fatal("GPU scorer retained CPU model")
	}
	if len(g.cpu.norm) != 0 || len(g.cpu.rope) != 0 {
		t.Fatal("GPU host retained CPU norm or rope")
	}
	if g.cpu.head != cpu.head {
		t.Fatal("GPU host did not reuse head")
	}
	if len(g.cpu.embedding) != len(cpu.embedding) {
		t.Fatal("GPU host embedding length mismatch")
	}
	if &g.cpu.embedding[0] != &cpu.embedding[0] {
		t.Fatal("GPU host embedding backing store changed")
	}
	if g.cpu.embedding[0] != cpu.embedding[0] || g.cpu.embedding[len(g.cpu.embedding)-1] != cpu.embedding[len(cpu.embedding)-1] {
		t.Fatal("GPU host embedding values changed")
	}
	if g.cpu.eps != cpu.eps || g.cpu.meta.VocabSize != cpu.meta.VocabSize || g.cpu.meta.HiddenSize != cpu.meta.HiddenSize {
		t.Fatal("GPU host metadata mismatch")
	}
	// Upload must not consume or mutate the caller's independent CPU scorer.
	got, err := cpu.ScoreEncoded(base.Row)
	if err != nil {
		t.Fatal(err)
	}
	checkNVIDIALifetimeLogits(t, got, base.Logits)
	runtime.KeepAlive(cpu)
	ok = true
	return g, w
}

func TestNVIDIATextScorerHostLifetime(t *testing.T) {
	dir, fixture, config := mojevNVIDIALifetimeCheckpoint(t)
	g, w := newNVIDIATextScorerLifetimeSubject(t, dir, config, fixture.Cases["base"])
	closed := false
	defer func() {
		if !closed && g != nil {
			_ = g.Close()
		}
	}()
	for i := 0; i < 64; i++ {
		runtime.GC()
		runtime.Gosched()
		if w.Value() == nil {
			break
		}
	}
	if w.Value() != nil {
		t.Fatal("CPU encoder model still reachable after GPU construction")
	}
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	t.Logf("heap_alloc=%d heap_inuse=%d heap_objects=%d", mem.HeapAlloc, mem.HeapInuse, mem.HeapObjects)
	got, err := g.ScoreEncoded(fixture.Cases["base"].Row)
	if err != nil {
		t.Fatal(err)
	}
	checkNVIDIALifetimeLogits(t, got, fixture.Cases["base"].Logits)
	host, head, embedding := weak.Make(g.cpu), weak.Make(g.cpu.head), weak.Make(&g.cpu.embedding[0])
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	if g.cpu != nil || g.ResidentBytes() != 0 {
		t.Fatal("closed scorer retained host view or device allocations")
	}
	for i := 0; i < 64; i++ {
		runtime.GC()
		runtime.Gosched()
		if host.Value() == nil && head.Value() == nil && embedding.Value() == nil {
			break
		}
	}
	if host.Value() != nil || head.Value() != nil || embedding.Value() != nil {
		t.Fatal("closed host view, head or embedding still reachable")
	}
	runtime.KeepAlive(g)
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
}

func checkNVIDIALifetimeLogits(t *testing.T, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("field count mismatch: got %d want %d", len(got), len(want))
	}
	var maxDiff float64
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("candidate count mismatch for field %d: got %d want %d", i, len(got[i]), len(want[i]))
		}
		for j, v := range got[i] {
			d := math.Abs(float64(v - want[i][j]))
			if d > maxDiff {
				maxDiff = d
			}
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-4 {
				t.Fatalf("base logits[%d][%d] got %.9g want %.9g diff %.6g", i, j, v, want[i][j], d)
			}
		}
	}
	t.Logf("base max logit error %g", maxDiff)
}
