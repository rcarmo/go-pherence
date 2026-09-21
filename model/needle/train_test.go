package needle

import (
	"math"
	"path/filepath"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func TestTrainingAndCheckpoint(t *testing.T) {
	m, f := fixture(t)
	opt := NewAdamW()
	initial := f.Loss
	var err error
	for i := 0; i < 8; i++ {
		m, _, err = m.TrainStep(opt, f.Tokens, nil, .003, Options{})
		if err != nil {
			t.Fatal(err)
		}
	}
	loss, _, err := m.LossGrad(f.Tokens, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if loss >= initial-.05 {
		t.Fatalf("loss did not decrease: %g -> %g", initial, loss)
	}
	path := filepath.Join(t.TempDir(), "trained.safetensors")
	if err = checkpoint.Save(path, m.Checkpoint()); err != nil {
		t.Fatal(err)
	}
	reloaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := m.Forward(f.Tokens, Options{})
	b, _ := reloaded.Forward(f.Tokens, Options{})
	compare(t, "checkpoint", a, b, 0, 0)
}
func TestLoRATraining(t *testing.T) {
	m, f := fixture(t)
	a, err := m.NewAdapter(2, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := m.Merge(a)
	if err != nil {
		t.Fatal(err)
	}
	base, _ := m.Forward(f.Tokens, Options{})
	same, _ := merged.Forward(f.Tokens, Options{})
	compare(t, "zero B", same, base, 0, 0)
	loss, g, err := m.AdapterLossGrad(a, f.Tokens, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// Check one adapter gradient against centered finite differences.
	name := "stack/layers/block/self_attn/out_proj/kernel"
	w := a.Weights[name]
	idx := 3
	eps := float32(.001)
	w.B.Data[idx] += eps
	plus, _, e := m.AdapterLossGrad(a, f.Tokens, nil, Options{})
	if e != nil {
		t.Fatal(e)
	}
	w.B.Data[idx] -= 2 * eps
	minus, _, e := m.AdapterLossGrad(a, f.Tokens, nil, Options{})
	if e != nil {
		t.Fatal(e)
	}
	w.B.Data[idx] += eps
	numeric := (plus - minus) / float64(2*eps)
	analytic := float64(g[name+"/B"].Data[idx])
	if math.Abs(numeric-analytic) > 2e-4 {
		t.Fatalf("LoRA gradient got %g numeric %g", analytic, numeric)
	}
	opt := NewAdamW()
	for i := 0; i < 12; i++ {
		a, _, err = m.TrainAdapterStep(a, opt, f.Tokens, nil, .005, Options{})
		if err != nil {
			t.Fatal(err)
		}
	}
	after, _, err := m.AdapterLossGrad(a, f.Tokens, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if after >= loss-.01 {
		t.Fatalf("LoRA did not improve loss %g -> %g", loss, after)
	}
	unchanged, _ := m.Forward(f.Tokens, Options{})
	compare(t, "base immutable", unchanged, base, 0, 0)
}
func TestOptimizerRejectsWithoutMutation(t *testing.T) {
	o := NewAdamW()
	w := map[string]checkpoint.Tensor{"p": {Shape: []int{1}, Data: []float32{1}}}
	g := map[string]checkpoint.Tensor{"p": {Shape: []int{1}, Data: []float32{float32(math.NaN())}}}
	if _, e := o.update(w, g, .01); e == nil {
		t.Fatal("accepted nan")
	}
	if o.step != 0 || w["p"].Data[0] != 1 {
		t.Fatal("mutated invalid update")
	}
	g["p"].Data[0] = 2
	u, e := o.update(w, g, .01)
	if e != nil {
		t.Fatal(e)
	}
	if math.Abs(float64(u["p"].Data[0]-.989999)) > 2e-6 {
		t.Fatalf("AdamW %v", u)
	}
}
func TestOwnedInputsAndConcurrentForward(t *testing.T) {
	m, f := fixture(t)
	cp := m.Checkpoint()
	for _, w := range cp.Tensors {
		for i := range w.Data {
			w.Data[i] = 99
		}
	}
	conf := m.Configuration()
	conf.EngramLayers[0] = 99
	for i := 0; i < 4; i++ {
		t.Run("parallel", func(t *testing.T) {
			t.Parallel()
			v, e := m.Forward(f.Tokens, Options{})
			if e != nil {
				t.Fatal(e)
			}
			compare(t, "immutable", v, f.Logits.Data, 3e-5, 3e-4)
		})
	}
}

func TestNeedle2Training(t *testing.T) {
	m, f := fixtureFile(t, "needle2.json")
	opt := NewAdamW()
	for i := 0; i < 6; i++ {
		var err error
		m, _, err = m.TrainStep(opt, f.Tokens, nil, .003, Options{})
		if err != nil {
			t.Fatal(err)
		}
	}
	loss, _, err := m.LossGrad(f.Tokens, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if loss >= f.Loss {
		t.Fatal("Needle2 training did not reduce loss")
	}
}

func TestPreservesAuxiliaryTensors(t *testing.T) {
	m, _ := fixture(t)
	cp := m.Checkpoint()
	cp.Tensors["router_head/calibration"] = checkpoint.Tensor{Shape: []int{3}, Data: []float32{.9, 0, .6}}
	m, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Checkpoint().Tensors["router_head/calibration"].Data) != 3 {
		t.Fatal("lost auxiliary tensor")
	}
}
