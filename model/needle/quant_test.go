package needle

import (
	"context"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"math"
	"slices"
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
	if _, err := v2.Forward(f.Tokens, Options{Quant: &Quantization{WeightBits: 4, KVBits: 8}}); err == nil {
		t.Fatal("advertised unsupported v2 simulated KV8")
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

func TestCQRejectsIncompleteABScales(t *testing.T) {
	m, _ := fixture(t)
	cp := m.Checkpoint()
	cp.Tensors["ab_scales/embedding/embedding/a"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	if _, err := New(cp); err == nil {
		t.Fatal("incomplete AB scales accepted")
	}
}

func TestA8TinyAndZeroRows(t *testing.T) {
	tp := &tape{train: true, limit: 1 << 20}
	x := tp.leaf([]float32{1e-8, -1e-8, 0})
	y := tp.fakeA8(x)
	compare(t, "tiny row", y.x, x.x, 1e-15, 0)
	for i := range y.g {
		y.g[i] = 1
	}
	tp.back()
	compare(t, "STE", x.g, []float32{1, 1, 1}, 0, 0)
	z := tp.fakeA8(tp.leaf([]float32{0, 0}))
	compare(t, "zero", z.x, []float32{0, 0}, 0, 0)
}

func TestNeedle2CQTrainingAndCache(t *testing.T) {
	m, f := fixtureFile(t, "needle2-cq.json")
	opts := Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8}}
	decoder, err := m.NewDecoder(DecoderOptions{Capacity: 16, Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range f.Tokens {
		got, err := decoder.Step(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.Forward(f.Tokens[:i+1], opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "v2 CQ cache", got, want[len(want)-m.config.OutVocab:], 3e-5, 1e-3)
	}
	a, err := m.Generate(context.Background(), f.Tokens, 8, -1, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.GenerateCached(context.Background(), f.Tokens, 8, -1, DecoderOptions{Execution: opts})
	if err != nil || !slices.Equal(a, b) {
		t.Fatalf("generated %v != %v: %v", a, b, err)
	}
	ad, err := m.NewAdapter(2, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	initial, _, err := m.AdapterLossGrad(ad, f.Tokens, nil, opts)
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
	after, _, err := m.AdapterLossGrad(ad, f.Tokens, nil, opts)
	if err != nil || after >= initial {
		t.Fatalf("adapter loss %g -> %g %v", initial, after, err)
	}
	next, initial, err := m.TrainStep(NewAdamW(), f.Tokens, nil, .001, opts)
	if err != nil {
		t.Fatal(err)
	}
	after, _, err = next.LossGrad(f.Tokens, nil, opts)
	if err != nil || after >= initial {
		t.Fatalf("trunk loss %g -> %g %v", initial, after, err)
	}
	for name, w := range f.Tensors {
		compare(t, "original "+name, m.tensors[name].Data, w.Data, 0, 0)
	}
}
