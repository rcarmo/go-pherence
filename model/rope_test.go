package model

import (
	"math"
	"testing"
)

func TestBuildRoPEFreqsRejectsOverflowAndBadDims(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got := buildRoPEFreqs(maxInt, 2, 4, 10000); got != nil {
		t.Fatalf("overflow maxSeq returned len=%d, want nil", len(got))
	}
	if got := buildRoPEFreqs(4, 2, 0, 10000); got != nil {
		t.Fatalf("zero headDim returned len=%d, want nil", len(got))
	}
	if got := buildRoPEFreqs(0, 2, 4, 10000); got != nil {
		t.Fatalf("zero maxSeq returned len=%d, want nil", len(got))
	}
}

func TestBuildRoPEFreqsNegativeThetaFallsBack(t *testing.T) {
	freqs := buildRoPEFreqs(2, 2, 4, -1)
	if len(freqs) != 8 {
		t.Fatalf("len=%d, want 8", len(freqs))
	}
	for i, v := range freqs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("freqs[%d]=%v, want finite", i, v)
		}
	}
}
