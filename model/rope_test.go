package model

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/model/common"
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

func TestGemma4FullAttentionRoPEUsesFullHeadFactors(t *testing.T) {
	m := &LlamaModel{Config: common.Config{ModelType: "gemma4_text", MaxSeqLen: 8, HeadDim: 256, GlobalHeadDim: 512}}
	m.precomputeGemma4RoPE()
	if m.RopeHalfSWA != 128 {
		t.Fatalf("RopeHalfSWA=%d, want 128", m.RopeHalfSWA)
	}
	if m.RopeHalfFull != 256 {
		t.Fatalf("RopeHalfFull=%d, want full-head half 256", m.RopeHalfFull)
	}
	if got, want := len(m.RopeFreqsFull), 8*256*2; got != want {
		t.Fatalf("full RoPE table len=%d, want %d", got, want)
	}
	factors := synthesizedGemma4FullAttentionRoPEFactors(512)
	if len(factors) != 256 {
		t.Fatalf("factors len=%d, want 256", len(factors))
	}
	for i := 0; i < 64; i++ {
		if factors[i] != 1 {
			t.Fatalf("factor[%d]=%g, want 1", i, factors[i])
		}
	}
	for i := 64; i < len(factors); i++ {
		if factors[i] < 1e20 {
			t.Fatalf("factor[%d]=%g, want disabled large factor", i, factors[i])
		}
	}
}
