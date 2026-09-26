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
	"github.com/rcarmo/go-pherence/tensor"
)

func mojevSIMDHostLifetimeCheckpoint(t *testing.T) (string, nativeTextFixture, []byte) {
	t.Helper()
	if os.Getenv("GO_PHERENCE_MOJEV_SIMD") != "1" {
		t.Skip("set GO_PHERENCE_MOJEV_SIMD=1")
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

type simdLifetimeWeakRefs struct {
	model        weak.Pointer[qwen.Qwen35BaseModel]
	linearTensor weak.Pointer[tensor.Tensor]
	linearData   weak.Pointer[float32]
	fullTensor   weak.Pointer[tensor.Tensor]
	fullData     weak.Pointer[float32]
}

func simdLifetimeTrackedProjections(t *testing.T, model *qwen.Qwen35BaseModel) (int, *tensor.Tensor, int, *tensor.Tensor) {
	t.Helper()
	if model == nil {
		t.Fatal("nil model")
	}
	linearIdx, fullIdx := -1, -1
	var linearTensor, fullTensor *tensor.Tensor
	for i, layer := range model.Layers {
		if linearTensor == nil && layer.Kind == qwen.Qwen35LinearAttentionLayerKind {
			if layer.Linear == nil || layer.Linear.QKVW == nil || len(layer.Linear.QKVW.Data()) == 0 {
				t.Fatal("missing first linear QKVW tensor")
			}
			linearIdx, linearTensor = i, layer.Linear.QKVW
		}
		if fullTensor == nil && layer.Kind == qwen.Qwen35FullAttentionLayerKind {
			if layer.Full == nil || layer.Full.QW == nil || len(layer.Full.QW.Data()) == 0 {
				t.Fatal("missing first full QW tensor")
			}
			fullIdx, fullTensor = i, layer.Full.QW
		}
		if linearTensor != nil && fullTensor != nil {
			break
		}
	}
	if linearTensor == nil || fullTensor == nil {
		t.Fatal("missing tracked projections")
	}
	return linearIdx, linearTensor, fullIdx, fullTensor
}

//go:noinline
func newSIMDTextScorerLifetimeSubject(t *testing.T, dir string, config []byte, base nativeTextCase) (*SIMDTextScorer, simdLifetimeWeakRefs) {
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
	if cpu == nil || cpu.model == nil || cpu.head == nil || len(cpu.embedding) == 0 || len(cpu.norm) == 0 || len(cpu.rope) == 0 {
		t.Fatal("invalid CPU scorer")
	}
	model := cpu.model
	linearIdx, linearTensor, fullIdx, fullTensor := simdLifetimeTrackedProjections(t, model)
	linearData, fullData := linearTensor.Data(), fullTensor.Data()
	linearFirst, linearLast := linearData[0], linearData[len(linearData)-1]
	fullFirst, fullLast := fullData[0], fullData[len(fullData)-1]
	refs := simdLifetimeWeakRefs{
		model:        weak.Make(model),
		linearTensor: weak.Make(linearTensor),
		linearData:   weak.Make(&linearData[0]),
		fullTensor:   weak.Make(fullTensor),
		fullData:     weak.Make(&fullData[0]),
	}
	s, err := NewSIMDTextScorer(cpu, 256)
	if err != nil {
		t.Fatal(err)
	}
	if s == nil || s.cpu == nil || s.branch == nil {
		t.Fatal("nil SIMD scorer")
	}
	if s.cpu == cpu {
		t.Fatal("SIMD scorer retained caller TextScorer")
	}
	if cpu.model != model {
		t.Fatal("constructor mutated caller model")
	}
	if cpu.model.Layers[linearIdx].Linear == nil || cpu.model.Layers[linearIdx].Linear.QKVW != linearTensor {
		t.Fatal("constructor replaced caller linear projection")
	}
	if cpu.model.Layers[fullIdx].Full == nil || cpu.model.Layers[fullIdx].Full.QW != fullTensor {
		t.Fatal("constructor replaced caller full projection")
	}
	if linearTensor.Data()[0] != linearFirst || linearTensor.Data()[len(linearTensor.Data())-1] != linearLast {
		t.Fatal("constructor mutated caller linear projection values")
	}
	if fullTensor.Data()[0] != fullFirst || fullTensor.Data()[len(fullTensor.Data())-1] != fullLast {
		t.Fatal("constructor mutated caller full projection values")
	}
	if s.cpu.model != nil {
		t.Fatal("SIMD scorer retained CPU model")
	}
	if len(s.cpu.norm) != len(cpu.norm) {
		t.Fatal("SIMD host norm length mismatch")
	}
	if &s.cpu.norm[0] != &cpu.norm[0] {
		t.Fatal("SIMD host norm backing store changed")
	}
	if s.cpu.norm[0] != cpu.norm[0] || s.cpu.norm[len(s.cpu.norm)-1] != cpu.norm[len(cpu.norm)-1] {
		t.Fatal("SIMD host norm values changed")
	}
	if len(s.cpu.rope) != len(cpu.rope) {
		t.Fatal("SIMD host rope length mismatch")
	}
	if &s.cpu.rope[0] != &cpu.rope[0] {
		t.Fatal("SIMD host rope backing store changed")
	}
	if s.cpu.rope[0] != cpu.rope[0] || s.cpu.rope[len(s.cpu.rope)-1] != cpu.rope[len(cpu.rope)-1] {
		t.Fatal("SIMD host rope values changed")
	}
	if s.cpu.head != cpu.head {
		t.Fatal("SIMD host did not reuse head")
	}
	if len(s.cpu.embedding) != len(cpu.embedding) {
		t.Fatal("SIMD host embedding length mismatch")
	}
	if &s.cpu.embedding[0] != &cpu.embedding[0] {
		t.Fatal("SIMD host embedding backing store changed")
	}
	if s.cpu.embedding[0] != cpu.embedding[0] || s.cpu.embedding[len(s.cpu.embedding)-1] != cpu.embedding[len(cpu.embedding)-1] {
		t.Fatal("SIMD host embedding values changed")
	}
	if s.cpu.eps != cpu.eps || s.cpu.meta.VocabSize != cpu.meta.VocabSize || s.cpu.meta.HiddenSize != cpu.meta.HiddenSize {
		t.Fatal("SIMD host metadata mismatch")
	}
	got, err := cpu.ScoreEncoded(base.Row)
	if err != nil {
		t.Fatal(err)
	}
	checkSIMDLifetimeLogits(t, got, base.Logits)
	runtime.KeepAlive(cpu)
	return s, refs
}

func simdLifetimeCleared(refs simdLifetimeWeakRefs) bool {
	return refs.model.Value() == nil && refs.linearTensor.Value() == nil && refs.linearData.Value() == nil && refs.fullTensor.Value() == nil && refs.fullData.Value() == nil
}

func TestSIMDTextScorerHostLifetime(t *testing.T) {
	dir, fixture, config := mojevSIMDHostLifetimeCheckpoint(t)
	s, refs := newSIMDTextScorerLifetimeSubject(t, dir, config, fixture.Cases["base"])
	for i := 0; i < 64; i++ {
		runtime.GC()
		runtime.Gosched()
		if simdLifetimeCleared(refs) {
			break
		}
	}
	if !simdLifetimeCleared(refs) {
		t.Fatal("original CPU model or projection roots still reachable after SIMD construction")
	}
	got, err := s.ScoreEncoded(fixture.Cases["base"].Row)
	if err != nil {
		t.Fatal(err)
	}
	checkSIMDLifetimeLogits(t, got, fixture.Cases["base"].Logits)
	runtime.KeepAlive(s)
}

func checkSIMDLifetimeLogits(t *testing.T, got, want [][]float32) {
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
