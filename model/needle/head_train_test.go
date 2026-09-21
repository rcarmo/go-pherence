package needle

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type headTrainRef struct {
	Target    []float32
	Loss      float64
	Output    checkpoint.Tensor
	Gradients map[string]checkpoint.Tensor
}
type headWidthRef struct {
	Heads map[HeadKind]map[string]headTrainRef
	Width struct {
		Config       json.RawMessage
		Tensors      map[string]checkpoint.Tensor
		Tokens       []int
		ChildConfig  json.RawMessage              `json:"child_config"`
		ChildTensors map[string]checkpoint.Tensor `json:"child_tensors"`
		Outputs      map[string]struct {
			Logits checkpoint.Tensor
			Heads  map[HeadKind]checkpoint.Tensor
		}
	}
}

func headWidthFixture(t *testing.T) headWidthRef {
	t.Helper()
	b, e := os.ReadFile("testdata/needle3-head-width.json")
	if e != nil {
		t.Fatal(e)
	}
	var f headWidthRef
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	return f
}
func TestHeadTrainingUpstream(t *testing.T) {
	m, ext := extended(t)
	f := headWidthFixture(t)
	for kind, modes := range f.Heads {
		for mode, want := range modes {
			t.Run(string(kind)+"/"+mode, func(t *testing.T) {
				opts := Options{}
				if mode == "cq" {
					opts.Quant = &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
				}
				loss, g, e := m.HeadLossGrad(ext.Base.Tokens, kind, want.Target, opts)
				if e != nil {
					t.Fatal(e)
				}
				if math.Abs(loss-want.Loss) > 1e-5 {
					t.Errorf("loss %g want %g", loss, want.Loss)
				}
				for key, w := range want.Gradients {
					got, ok := g[key]
					if !ok {
						for _, v := range w.Data {
							if v != 0 {
								t.Errorf("missing nonzero gradient %s", key)
								break
							}
						}
						continue
					}
					compare(t, key, got.Data, w.Data, 2e-5, 8e-3)
					if !strings.HasPrefix(key, string(kind)+"_head/") {
						t.Fatal("trunk gradient leaked")
					}
				}
			})
		}
	}
}
func TestHeadTrainUpdatesOnlyHead(t *testing.T) {
	base, ext := extended(t)
	f := headWidthFixture(t)
	for kind, modes := range f.Heads {
		m := base
		opt := NewAdamW()
		target := modes["fp32"].Target
		initial, _, e := m.HeadLossGrad(ext.Base.Tokens, kind, target, Options{})
		if e != nil {
			t.Fatal(e)
		}
		for i := 0; i < 6; i++ {
			m, _, e = m.TrainHeadStep(opt, ext.Base.Tokens, kind, target, .003, Options{})
			if e != nil {
				t.Fatal(e)
			}
		}
		after, _, e := m.HeadLossGrad(ext.Base.Tokens, kind, target, Options{})
		if e != nil {
			t.Fatal(e)
		}
		if after >= initial {
			t.Errorf("%s no loss reduction %g -> %g", kind, initial, after)
		}
		for key, w := range base.tensors {
			if !strings.HasPrefix(key, string(kind)+"_head/") && !slices.Equal(m.tensors[key].Data, w.Data) {
				t.Fatalf("mutated %s", key)
			}
		}
	}
}
func TestHeadTrainAdmission(t *testing.T) {
	m, f := extended(t)
	for _, tc := range []struct {
		k HeadKind
		t []float32
	}{{Confidence, []float32{2}}, {Confidence, nil}, {Router, []float32{1.5}}, {Embedding, []float32{0, 0, 0, 0}}, {Embedding, []float32{1}}, {Router, []float32{float32(math.NaN())}}} {
		if _, _, err := m.HeadLossGrad(f.Base.Tokens, tc.k, tc.t, Options{}); err == nil {
			t.Fatalf("accepted %v", tc)
		}
	}
	if _, _, err := m.HeadLossGrad(f.Base.Tokens, Confidence, []float32{1}, Options{MaxWorkBytes: 16}); err == nil {
		t.Fatal("ignored budget")
	}
	a, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = a.HeadLossGrad(f.Base.Tokens, Confidence, []float32{1}, Options{}); err == nil {
		t.Fatal("archive head training admitted")
	}
}
func BenchmarkNeedleHeadGradient(b *testing.B) {
	m, f := extended(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, e := m.HeadLossGrad(f.Base.Tokens, Confidence, []float32{1}, Options{}); e != nil {
			b.Fatal(e)
		}
	}
}

func TestHeadTrainingCheckpointRoundtrip(t *testing.T) {
	m, f := extended(t)
	trained, _, err := m.TrainHeadStep(NewAdamW(), f.Base.Tokens, Confidence, []float32{1}, .003, Options{})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "trained.safetensors")
	if err = checkpoint.Save(path, trained.Checkpoint()); err != nil {
		t.Fatal(err)
	}
	reloaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := trained.Head(f.Base.Tokens, Confidence, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := reloaded.Head(f.Base.Tokens, Confidence, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "head save/load", got, want, 0, 0)
}
