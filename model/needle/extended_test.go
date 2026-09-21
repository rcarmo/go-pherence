package needle

import (
	"encoding/json"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"math"
	"os"
	"slices"
	"testing"
)

type headOutputs struct {
	Embedding, Router []float32
	Confidence        float32
}
type outputFixture struct {
	Logits checkpoint.Tensor
	Heads  headOutputs
}
type sliceFixture struct {
	Config  json.RawMessage
	Tensors map[string]checkpoint.Tensor
	Tokens  []int
	outputFixture
}
type extendedFixture struct {
	Base struct {
		Config   json.RawMessage
		Tensors  map[string]checkpoint.Tensor
		Tokens   []int
		FP32, CQ outputFixture
		Slices   map[string]sliceFixture
	}
	AB reference
}

func extended(t testing.TB) (*Model, extendedFixture) {
	t.Helper()
	b, err := os.ReadFile("testdata/needle3-extended.json")
	if err != nil {
		t.Fatal(err)
	}
	var f extendedFixture
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	m, err := New(&checkpoint.Checkpoint{Config: f.Base.Config, FormatVersion: 2, Tensors: f.Base.Tensors})
	if err != nil {
		t.Fatal(err)
	}
	return m, f
}
func checkHeads(t *testing.T, m *Model, ids []int, opts Options, want headOutputs) {
	t.Helper()
	for kind, w := range map[HeadKind][]float32{Embedding: want.Embedding, Confidence: {want.Confidence}, Router: want.Router} {
		o, err := m.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, string(kind), o, w, 1e-5, 3e-3)
	}
}
func TestNeedle3HeadsUpstream(t *testing.T) {
	m, f := extended(t)
	for _, cq := range []bool{false, true} {
		opts := Options{}
		want := f.Base.FP32
		if cq {
			opts.Quant = &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
			want = f.Base.CQ
		}
		logits, err := m.Forward(f.Base.Tokens, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "extended logits", logits, want.Logits.Data, 5e-5, 1e-3)
		checkHeads(t, m, f.Base.Tokens, opts, want.Heads)
	}
}
func TestNeedle3LadderUpstream(t *testing.T) {
	m, f := extended(t)
	for name, ref := range f.Base.Slices {
		t.Run(name, func(t *testing.T) {
			parent := m
			if name == "nested3then2" {
				var err error
				parent, err = m.SliceDepth(3)
				if err != nil {
					t.Fatal(err)
				}
			}
			var cfg Config
			if err := json.Unmarshal(ref.Config, &cfg); err != nil {
				t.Fatal(err)
			}
			child, err := parent.SliceDepth(cfg.Layers)
			if err != nil {
				t.Fatal(err)
			}
			for key, w := range ref.Tensors {
				got, ok := child.tensors[key]
				if !ok || !slices.Equal(got.Shape, w.Shape) {
					t.Fatalf("slice tensor %s", key)
				}
				compare(t, key, got.Data, w.Data, 0, 0)
			}
			out, err := child.Forward(ref.Tokens, Options{})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "rung logits", out, ref.Logits.Data, 3e-5, 3e-4)
			checkHeads(t, child, ref.Tokens, Options{}, ref.Heads)
		})
	}
}
func TestNeedle3ABUpstream(t *testing.T) {
	_, f := extended(t)
	m, err := New(&checkpoint.Checkpoint{Config: f.AB.Config, FormatVersion: 2, Tensors: f.AB.Tensors})
	if err != nil {
		t.Fatal(err)
	}
	opts := Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}}
	out, err := m.Forward(f.AB.Tokens, opts)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "AB logits", out, f.AB.Logits.Data, 8e-5, 1e-3)
	loss, g, err := m.LossGrad(f.AB.Tokens, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loss-f.AB.Loss) > 8e-6 {
		t.Errorf("AB loss %g want %g", loss, f.AB.Loss)
	}
	for key, w := range f.AB.Gradients {
		got, ok := g[key]
		if !ok {
			for _, v := range w.Data {
				if v != 0 {
					t.Errorf("missing gradient %s", key)
					break
				}
			}
			continue
		}
		compare(t, key, got.Data, w.Data, 1e-5, 1e-2)
	}
}
func TestHeadAdmission(t *testing.T) {
	m, f := extended(t)
	for _, ids := range [][]int{nil, {0, 0}, {-1}, {16}} {
		if _, err := m.Head(ids, Embedding, Options{}); err == nil {
			t.Errorf("accepted %v", ids)
		}
	}
	if _, err := m.Head(f.Base.Tokens, HeadKind("other"), Options{}); err == nil {
		t.Fatal("accepted unknown head")
	}
	if _, err := m.Head(f.Base.Tokens, Embedding, Options{MaxWorkBytes: 128}); err == nil {
		t.Fatal("ignored cap")
	}
	cp := m.Checkpoint()
	delete(cp.Tensors, "embedding_head/query")
	m, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Head(f.Base.Tokens, Embedding, Options{}); err == nil {
		t.Fatal("missing query accepted")
	}
}

func TestABAdmission(t *testing.T) {
	m, _ := fixture(t)
	for _, scales := range []map[string]checkpoint.Tensor{
		{"ab_scales/embedding/embedding/a": {Shape: []int{1}, Data: []float32{1}}},
		{"ab_scales/no_such/kernel/a": {Shape: []int{1}, Data: []float32{1}}, "ab_scales/no_such/kernel/b": {Shape: []int{1}, Data: []float32{1}}},
		{"ab_scales/embedding/embedding/a": {Shape: []int{3}, Data: []float32{1, 1, 1}}, "ab_scales/embedding/embedding/b": {Shape: []int{1}, Data: []float32{1}}},
	} {
		cp := m.Checkpoint()
		for key, v := range scales {
			cp.Tensors[key] = v
		}
		if _, err := New(cp); err == nil {
			t.Fatal("invalid AB accepted")
		}
	}
}

func TestHeadMasksQKVCurrentTap(t *testing.T) {
	m, _ := fixture(t)
	e := m.execution(false, Options{})
	e.keep = []bool{true, false, true}
	a := e.t.leaf([]float32{1, 2, 3})
	a.r, a.c = 3, 1
	taps := e.t.leaf([]float32{1, .5})
	taps.r, taps.c = 2, 1
	// Attention taps do not mask offset zero, even on a pad query; engram taps do.
	compare(t, "QKV pad", e.conv(a, taps, 1, 0, false).x, []float32{1, 2.5, 3}, 0, 0)
	compare(t, "engram pad", e.conv(a, taps, 1, 0, true).x, []float32{1, .5, 3}, 0, 0)
}

func TestHeadFullyMaskedRowMatchesUpstreamFiniteMask(t *testing.T) {
	m, _ := fixture(t)
	e := m.execution(false, Options{})
	e.keep = []bool{false, true}
	a := e.t.leaf([]float32{2, 3, 4, 5})
	a.r, a.c = 2, 2
	got := e.attentionSoftmax(a, 0)
	compare(t, "finite masked row", got.x, []float32{.5, .5, 0, 1}, 0, 0)
}
func TestNilModelErrors(t *testing.T) {
	var m *Model
	if _, err := m.Forward([]int{2}, Options{}); err == nil {
		t.Fatal("nil forward")
	}
	if _, _, err := m.LossGrad([]int{2, 3}, nil, Options{}); err == nil {
		t.Fatal("nil loss")
	}
	if _, err := m.Head([]int{2}, Embedding, Options{}); err == nil {
		t.Fatal("nil head")
	}
}
