package simd

import (
	"math"
	"testing"
)

func TestGQAAttentionScaleToMatchesAllocated(t *testing.T) {
	seqLen, numHeads, numKVHeads, headDim := 3, 2, 1, 2
	q := []float32{1, 0.5, -0.5, 1}
	k := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	v := []float32{
		1, 2,
		3, 4,
		-1, 0.5,
	}
	want := GQAAttentionScale(q, k, v, seqLen, numHeads, numKVHeads, headDim, 0.5)
	got := make([]float32, len(want)+1)
	scores := make([]float32, seqLen)
	got[len(got)-1] = 123
	if !GQAAttentionScaleTo(got, scores, q, k, v, seqLen, numHeads, numKVHeads, headDim, 0.5) {
		t.Fatal("GQAAttentionScaleTo returned false")
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatal("GQAAttentionScaleTo mutated tail")
	}
}

func TestGQAAttentionScaleChecked(t *testing.T) {
	seqLen, numHeads, numKVHeads, headDim := 2, 2, 1, 2
	q := []float32{1, 0, 0, 1}
	kv := []float32{1, 2, 3, 4}
	got, ok := GQAAttentionScaleChecked(q, kv, kv, seqLen, numHeads, numKVHeads, headDim, 1)
	if !ok || len(got) != numHeads*headDim {
		t.Fatalf("GQAAttentionScaleChecked len=%d ok=%v", len(got), ok)
	}
	if got, ok := GQAAttentionScaleChecked(q[:3], kv, kv, seqLen, numHeads, numKVHeads, headDim, 1); ok || got != nil {
		t.Fatalf("GQAAttentionScaleChecked accepted short q: %v %v", got, ok)
	}
	zero, ok := GQAAttentionScaleChecked(q, nil, nil, 0, numHeads, numKVHeads, headDim, 1)
	if !ok || len(zero) != numHeads*headDim {
		t.Fatalf("GQAAttentionScaleChecked zero seq len=%d ok=%v", len(zero), ok)
	}
}

func TestGQAAttentionScaleToRejectsMalformedInputs(t *testing.T) {
	seqLen, numHeads, numKVHeads, headDim := 2, 2, 1, 2
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	q := make([]float32, numHeads*headDim)
	kv := make([]float32, seqLen*numKVHeads*headDim)
	if GQAAttentionScaleTo(out, scores, q, kv, kv, 0, numHeads, numKVHeads, headDim, 1) {
		t.Fatal("accepted zero seqLen")
	}
	if GQAAttentionScaleTo(out, scores, q, kv, kv, seqLen, 3, 2, headDim, 1) {
		t.Fatal("accepted non-divisible head grouping")
	}
	if GQAAttentionScaleTo(out[:3], scores, q, kv, kv, seqLen, numHeads, numKVHeads, headDim, 1) {
		t.Fatal("accepted short out")
	}
	if GQAAttentionScaleTo(out, scores[:1], q, kv, kv, seqLen, numHeads, numKVHeads, headDim, 1) {
		t.Fatal("accepted short scores")
	}
	if GQAAttentionScaleTo(out, scores, q[:3], kv, kv, seqLen, numHeads, numKVHeads, headDim, 1) {
		t.Fatal("accepted short q")
	}
	if GQAAttentionScaleTo(out, scores, q, kv[:3], kv, seqLen, numHeads, numKVHeads, headDim, 1) {
		t.Fatal("accepted short k cache")
	}
	maxInt := int(^uint(0) >> 1)
	if GQAAttentionScaleTo(out, scores, q, kv, kv, maxInt/2+1, 3, 1, 2, 1) {
		t.Fatal("accepted overflowing dimensions")
	}
}
