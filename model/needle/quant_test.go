package needle

import (
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"math"
	"testing"
)

func TestQuantizedTraining(t *testing.T) {
	m, f := fixture(t)
	opts := Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}}
	initial, _, err := m.LossGrad(f.Tokens, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	ad, err := m.NewAdapter(2, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	opt := NewAdamW()
	for i := 0; i < 12; i++ {
		ad, _, err = m.TrainAdapterStep(ad, opt, f.Tokens, nil, .003, opts)
		if err != nil {
			t.Fatal(err)
		}
	}
	loss, _, err := m.AdapterLossGrad(ad, f.Tokens, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if loss >= initial {
		t.Fatalf("QAT LoRA loss %g -> %g", initial, loss)
	}
}
func TestQuantizationAdmission(t *testing.T) {
	m, f := fixture(t)
	for _, q := range []*Quantization{{WeightBits: 3}, {ActivationBits: 4}, {KVBits: 4}, {WeightBits: math.NaN()}} {
		if _, err := m.Forward(f.Tokens, Options{Quant: q}); err == nil {
			t.Fatalf("accepted %+v", q)
		}
	}
	v2, _ := fixtureFile(t, "needle2.json")
	if _, err := v2.Forward(f.Tokens, Options{Quant: &Quantization{WeightBits: 4}}); err == nil {
		t.Fatal("advertised unsupported v2 CQ")
	}
}
func TestCQCodebooks(t *testing.T) {
	for _, key := range []string{"1", "1.58", "2", "4", "8"} {
		cb := cqCodebooks[key]
		if len(cb) < 2 {
			t.Fatalf("missing %s", key)
		}
		for i := 1; i < len(cb); i++ {
			if cb[i] <= cb[i-1] {
				t.Fatal("unordered codebook")
			}
		}
	}
	if nearestFP16(1+1.0/2048) != 1 {
		t.Fatal("FP16 midpoint must tie to even")
	}
	if nearestFP16(1+3.0/2048) != 1+2.0/1024 {
		t.Fatal("FP16 odd midpoint must round up")
	}
}

func TestCQRejectsABScales(t *testing.T) {
	m, f := fixture(t)
	cp := m.Checkpoint()
	cp.Tensors["ab_scales/embedding/embedding/a"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	m, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Forward(f.Tokens, Options{Quant: &Quantization{WeightBits: 4}}); err == nil {
		t.Fatal("AB scales silently ignored")
	}
}
