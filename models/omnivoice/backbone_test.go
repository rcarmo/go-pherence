package omnivoice

import (
	"context"
	"encoding/json"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"os"
	"testing"
)

type backboneFixture struct {
	Tokens    int       `json:"tokens"`
	IDs       []int     `json:"ids"`
	AudioMask []bool    `json:"audio_mask"`
	Logits    []float32 `json:"logits"`
}

func loadBackboneFixture(t *testing.T) (*Backbone, backboneFixture) {
	t.Helper()
	path := "../../testdata/omnivoice/backbone"
	raw, err := os.ReadFile(path + "/expected.json")
	if err != nil {
		t.Fatal(err)
	}
	var f backboneFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	w, err := loader.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	b, err := NewBackbone(w, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	return b, f
}
func TestBackboneUpstreamParity(t *testing.T) {
	b, f := loadBackboneFixture(t)
	out := make([]float32, len(f.Logits))
	for step := 0; step < 3; step++ {
		if err := b.ForwardInto(context.Background(), out, f.IDs, f.AudioMask, nil, nil); err != nil {
			t.Fatal(err)
		}
		maxError := float64(0)
		for i, v := range out {
			diff := math.Abs(float64(v - f.Logits[i]))
			maxError = math.Max(maxError, diff)
			if diff > 3e-5 || math.IsNaN(diff) {
				t.Fatalf("index %d got %g want %g", i, v, f.Logits[i])
			}
		}
		t.Logf("max logit error %g", maxError)
	}
}
func TestBackboneZeroAllocations(t *testing.T) {
	b, f := loadBackboneFixture(t)
	out := make([]float32, len(f.Logits))
	var err error
	n := testing.AllocsPerRun(10, func() { err = b.ForwardInto(context.Background(), out, f.IDs, f.AudioMask, nil, nil) })
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("allocations %g", n)
	}
}
func TestBackboneValidation(t *testing.T) {
	b, f := loadBackboneFixture(t)
	out := make([]float32, len(f.Logits))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := b.ForwardInto(ctx, out, f.IDs, f.AudioMask, nil, nil); err != context.Canceled {
		t.Fatal(err)
	}
	ids := append([]int(nil), f.IDs...)
	ids[1] = 99
	if err := b.ForwardInto(context.Background(), out, ids, f.AudioMask, nil, nil); err == nil {
		t.Fatal("invalid audio token accepted")
	}
	if err := b.ForwardInto(context.Background(), out[:1], f.IDs, f.AudioMask, nil, nil); err == nil {
		t.Fatal("bad output shape accepted")
	}
}
