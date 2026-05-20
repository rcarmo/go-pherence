package simd

import (
	"math"
	"testing"
)

func TestBuildRoPEFreqsRejectsOverflowAndBadDims(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got := BuildRoPEFreqs(maxInt, 2, 4, 10000); got != nil {
		t.Fatalf("overflow maxSeq returned len=%d, want nil", len(got))
	}
	if got := BuildRoPEFreqs(4, 2, 0, 10000); got != nil {
		t.Fatalf("zero headDim returned len=%d, want nil", len(got))
	}
	if got := BuildRoPEFreqs(0, 2, 4, 10000); got != nil {
		t.Fatalf("zero maxSeq returned len=%d, want nil", len(got))
	}
	if _, ok := CheckedRoPEFreqLen(maxInt/2+1, 2); ok {
		t.Fatal("CheckedRoPEFreqLen accepted overflowing dimensions")
	}
}

func TestBuildRoPEFreqsNegativeThetaFallsBack(t *testing.T) {
	freqs := BuildRoPEFreqs(2, 2, 4, -1)
	if len(freqs) != 8 {
		t.Fatalf("len=%d, want 8", len(freqs))
	}
	for i, v := range freqs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("freqs[%d]=%v, want finite", i, v)
		}
	}
}
