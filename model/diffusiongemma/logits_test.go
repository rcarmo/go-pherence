package diffusiongemma

import (
	"math"
	"testing"
)

func TestFinalLogitSoftcapPreservesSparseNegativeInfinity(t *testing.T) {
	scratch := ForwardScratch{FinalLogitSoftcapping: 30, Logits: [][]float32{{0, float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.NaN())}}}
	applyFinalLogitSoftcapping(scratch, 1, 4)
	if scratch.Logits[0][0] != 0 {
		t.Fatalf("zero logit changed to %v", scratch.Logits[0][0])
	}
	if !math.IsInf(float64(scratch.Logits[0][1]), -1) {
		t.Fatalf("sparse -Inf sentinel was not preserved: %v", scratch.Logits[0][1])
	}
	if scratch.Logits[0][2] != 30 {
		t.Fatalf("+Inf softcap=%v want 30", scratch.Logits[0][2])
	}
	if !math.IsNaN(float64(scratch.Logits[0][3])) {
		t.Fatalf("NaN logit was not preserved: %v", scratch.Logits[0][3])
	}
}
