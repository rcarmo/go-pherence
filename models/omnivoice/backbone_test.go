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

func sliceBookMajor(ids []int, books, fromTokens, toTokens int) []int {
	out := make([]int, books*toTokens)
	for book := 0; book < books; book++ {
		copy(out[book*toTokens:(book+1)*toTokens], ids[book*fromTokens:book*fromTokens+toTokens])
	}
	return out
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

func TestBackboneReconfigureParityAndBounds(t *testing.T) {
	b, f := loadBackboneFixture(t)
	books := len(f.IDs) / f.Tokens
	vocab := b.weights.Config.AudioVocabSize
	w2, err := loader.OpenWeights("../../testdata/omnivoice/backbone")
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	independent, err := NewBackbone(w2, 2)
	if err != nil {
		t.Fatal(err)
	}
	ids2 := sliceBookMajor(f.IDs, books, f.Tokens, 2)
	audio2 := append([]bool(nil), f.AudioMask[:2]...)
	want2 := make([]float32, books*2*vocab)
	if err = independent.ForwardInto(context.Background(), want2, ids2, audio2, nil, nil); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		tokens int
		ids    []int
		audio  []bool
		want   []float32
	}{
		{tokens: f.Tokens, ids: f.IDs, audio: f.AudioMask, want: f.Logits},
		{tokens: 2, ids: ids2, audio: audio2, want: want2},
		{tokens: f.Tokens, ids: f.IDs, audio: f.AudioMask, want: f.Logits},
	}
	for i, tc := range cases {
		if err := b.Reconfigure(tc.tokens); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(tc.want))
		if err := b.ForwardInto(context.Background(), got, tc.ids, tc.audio, nil, nil); err != nil {
			t.Fatal(err)
		}
		for j, v := range got {
			if diff := math.Abs(float64(v - tc.want[j])); diff > 3e-5 || math.IsNaN(diff) {
				t.Fatalf("case %d index %d got %g want %g", i, j, v, tc.want[j])
			}
		}
	}
	if err := b.Reconfigure(0); err == nil {
		t.Fatal("zero tokens accepted")
	}
	if err := b.Reconfigure(f.Tokens + 1); err == nil {
		t.Fatal("tokens above capacity accepted")
	}
	out2 := make([]float32, len(want2))
	out3 := make([]float32, len(f.Logits))
	var allocErr error
	n := testing.AllocsPerRun(10, func() {
		if allocErr = b.Reconfigure(2); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), out2, ids2, audio2, nil, nil)
		if allocErr != nil {
			return
		}
		if allocErr = b.Reconfigure(f.Tokens); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), out3, f.IDs, f.AudioMask, nil, nil)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if n != 0 {
		t.Fatalf("reuse allocations %g", n)
	}
}

// Deterministic cancellation after work has started, without wall-clock races.
type cancelAfterChecks struct {
	context.Context
	remaining int
}

func (c *cancelAfterChecks) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func TestBackboneCancelDuringAndReuse(t *testing.T) {
	b, f := loadBackboneFixture(t)
	out := make([]float32, len(f.Logits))
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 3}
	if err := b.ForwardInto(ctx, out, f.IDs, f.AudioMask, nil, nil); err != context.Canceled {
		t.Fatalf("expected cancellation, got %v", err)
	}
	if err := b.ForwardInto(context.Background(), out, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	for i := range out {
		if math.Abs(float64(out[i]-f.Logits[i])) > 3e-5 {
			t.Fatalf("retry mismatch %d", i)
		}
	}
}
