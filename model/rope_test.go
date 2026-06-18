package model

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
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

func TestGemma4RoPEUsesLocalGGMLThetaProgression(t *testing.T) {
	maxSeq, headDim := 1024, 512
	factors := synthesizedGemma4FullAttentionRoPEFactors(headDim)
	got := buildGemma4GGMLRoPEFreqsWithFactors(maxSeq, headDim/2, headDim, 1000000, factors)
	generic := simd.BuildRoPEFreqsWithFactors(maxSeq, headDim/2, headDim, 1000000, factors)
	if len(got) != len(generic) || len(got) == 0 {
		t.Fatalf("bad RoPE table lens got=%d generic=%d", len(got), len(generic))
	}
	// High positions make ggml's float32 iterative theta progression diverge from
	// the shared generic pow-based table. Keep this local to Gemma4 so other model
	// families retain the shared runtime builder they were validated against.
	idx := ((maxSeq-1)*(headDim/2) + 63) * 2
	if got[idx] == generic[idx] && got[idx+1] == generic[idx+1] {
		t.Fatalf("Gemma4 local RoPE unexpectedly matches shared generic table at idx=%d", idx)
	}
}

func TestGemma4RealGGUFFullAttentionRoPEFactors(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	tensor, ok := g.TensorByName("rope_freqs.weight")
	if !ok {
		t.Fatal("missing rope_freqs.weight")
	}
	factors, err := g.DequantF32(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if len(factors) != 256 {
		t.Fatalf("rope_freqs.weight len=%d want 256", len(factors))
	}
	for i := 0; i < 64; i++ {
		if factors[i] != 1 {
			t.Fatalf("rope_freqs[%d]=%g, want active factor 1", i, factors[i])
		}
	}
	for i := 64; i < len(factors); i++ {
		if factors[i] < 1e20 {
			t.Fatalf("rope_freqs[%d]=%g, want disabled large factor", i, factors[i])
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
