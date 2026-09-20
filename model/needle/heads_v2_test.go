package needle

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type v2HeadFixture struct {
	Pin     string                       `json:"upstream_pin"`
	Config  json.RawMessage              `json:"config"`
	Tokens  []int                        `json:"tokens"`
	Padded  []int                        `json:"padded_tokens"`
	Tensors map[string]checkpoint.Tensor `json:"tensors"`
	Heads   map[HeadKind]struct {
		Target       []float32                    `json:"target"`
		Loss         float64                      `json:"loss"`
		Output       checkpoint.Tensor            `json:"output"`
		PaddedOutput checkpoint.Tensor            `json:"padded_output"`
		Gradients    map[string]checkpoint.Tensor `json:"gradients"`
	} `json:"heads"`
}

func loadV2Heads(t *testing.T) (*Model, v2HeadFixture) {
	t.Helper()
	var f v2HeadFixture
	b, err := os.ReadFile("testdata/needle2-heads.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if f.Pin != "741ee892c5f8c4f5c0bb467c9566ea7a1eba919b" {
		t.Fatal("pin changed")
	}
	m, err := New(&checkpoint.Checkpoint{FormatVersion: 2, Config: f.Config, Tensors: f.Tensors})
	if err != nil {
		t.Fatal(err)
	}
	return m, f
}
func TestNeedle2HeadsUpstream(t *testing.T) {
	m, f := loadV2Heads(t)
	for kind, want := range f.Heads {
		t.Run(string(kind), func(t *testing.T) {
			got, err := m.Head(f.Tokens, kind, Options{})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "output", got, want.Output.Data, 1e-5, 3e-3)
			got, err = m.Head(f.Padded, kind, Options{})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "padded", got, want.PaddedOutput.Data, 1e-5, 3e-3)
			loss, grads, err := m.HeadLossGrad(f.Tokens, kind, want.Target, Options{})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "loss", []float32{float32(loss)}, []float32{float32(want.Loss)}, 1e-5, 1e-4)
			prefix := string(kind) + "_head/"
			for name, w := range want.Gradients {
				if !strings.HasPrefix(name, prefix) {
					continue
				}
				if name == "contrastive_head/log_temp" {
					compare(t, "unused temperature gradient", w.Data, []float32{0}, 0, 0)
					continue
				}
				got, ok := grads[name]
				if !ok {
					t.Fatal("missing gradient " + name)
				}
				compare(t, name, got.Data, w.Data, 1e-5, 4e-3)
			}
			next, _, err := m.TrainHeadStep(NewAdamW(), f.Tokens, kind, want.Target, 1e-3, Options{})
			if err != nil {
				t.Fatal(err)
			}
			after, _, err := next.HeadLossGrad(f.Tokens, kind, want.Target, Options{})
			if err != nil || after >= loss {
				t.Fatalf("loss %g -> %g %v", loss, after, err)
			}
			for name, v := range m.tensors {
				if !strings.HasPrefix(name, prefix) || name == "contrastive_head/log_temp" {
					compare(t, "frozen "+name, next.tensors[name].Data, v.Data, 0, 0)
				}
			}
			path := filepath.Join(t.TempDir(), "v2-head.safetensors")
			if err = checkpoint.Save(path, next.Checkpoint()); err != nil {
				t.Fatal(err)
			}
			loaded, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			got, err = loaded.Head(f.Tokens, kind, Options{})
			if err != nil {
				t.Fatal(err)
			}
			expected, err := next.Head(f.Tokens, kind, Options{})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "reload", got, expected, 0, 0)
		})
	}
}
func TestNeedle2HeadAdmission(t *testing.T) {
	m, f := loadV2Heads(t)
	for _, kind := range []HeadKind{Embedding, Router, "unknown"} {
		if _, err := m.Head(f.Tokens, kind, Options{}); err == nil {
			t.Fatal("unsupported head")
		}
	}
	for _, kind := range []HeadKind{Contrastive, Confidence} {
		if _, err := m.Head([]int{0, 0}, kind, Options{}); err == nil {
			t.Fatal("padding")
		}
		if _, err := m.Head(f.Tokens, kind, Options{Quant: &Quantization{WeightBits: 4}}); err == nil {
			t.Fatal("v2 CQ admitted")
		}
		cp := m.Checkpoint()
		delete(cp.Tensors, string(kind)+"_head/probes")
		bad, err := New(cp)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = bad.Head(f.Tokens, kind, Options{}); err == nil {
			t.Fatal("missing head")
		}
	}
	if _, _, err := m.HeadLossGrad(f.Tokens, Contrastive, []float32{2, 0, 0, 0}, Options{}); err == nil {
		t.Fatal("unnormalized target")
	}
}

func TestNeedle2HeadShapeRejection(t *testing.T) {
	m, f := loadV2Heads(t)
	for _, kind := range []HeadKind{Contrastive, Confidence} {
		for _, name := range []string{"probes", "proj/kernel"} {
			cp := m.Checkpoint()
			key := string(kind) + "_head/" + name
			old := cp.Tensors[key]
			old.Shape = []int{len(old.Data)}
			cp.Tensors[key] = old
			bad, err := New(cp)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = bad.Head(f.Tokens, kind, Options{}); err == nil {
				t.Fatal("malformed shape accepted: " + key)
			}
		}
	}
	cp := m.Checkpoint()
	delete(cp.Tensors, "contrastive_head/log_temp")
	bad, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = bad.Head(f.Tokens, Contrastive, Options{}); err == nil {
		t.Fatal("missing temperature accepted")
	}
	v3, _ := fixtureFile(t, "needle3.json")
	if _, err = v3.Head(f.Tokens, Contrastive, Options{}); err == nil {
		t.Fatal("v3 contrastive accepted")
	}
}
