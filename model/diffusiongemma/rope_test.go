package diffusiongemma

import (
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestSynthesizedFullAttentionRoPEFactors(t *testing.T) {
	factors := synthesizedFullAttentionRoPEFactors(512)
	if len(factors) != 256 {
		t.Fatalf("len=%d want 256", len(factors))
	}
	for i := 0; i < 64; i++ {
		if factors[i] != 1 {
			t.Fatalf("factor[%d]=%g want 1", i, factors[i])
		}
	}
	for i := 64; i < len(factors); i++ {
		if factors[i] != diffusionGemmaSuppressedRoPEFactor {
			t.Fatalf("factor[%d]=%g want %g", i, factors[i], diffusionGemmaSuppressedRoPEFactor)
		}
	}
}

func TestBuildRoPEPlanUsesSynthesizedFullAttentionFactors(t *testing.T) {
	plan := BuildRoPEPlan(Shape{CanvasLength: 2, TextHeadDim: 256, TextGlobalHeadDim: 512})
	plain := simd.BuildRoPEFreqs(2, 256, 512, 1000000)
	if len(plan.FullFreqs) != len(plain) {
		t.Fatalf("FullFreqs len=%d want %d", len(plan.FullFreqs), len(plain))
	}
	// Pair 0 is active, so it matches plain RoPE.
	if plan.FullFreqs[0] != plain[0] || plan.FullFreqs[1] != plain[1] {
		t.Fatalf("active pair mismatch got=(%g,%g) want=(%g,%g)", plan.FullFreqs[0], plan.FullFreqs[1], plain[0], plain[1])
	}
	// Pair 64 is suppressed by the synthesized 1e30 factor, so at pos=1 its
	// angle is effectively zero and differs from plain proportional-free RoPE.
	idx := (1*256 + 64) * 2
	if plan.FullFreqs[idx] == plain[idx] && plan.FullFreqs[idx+1] == plain[idx+1] {
		t.Fatalf("suppressed pair unexpectedly matches plain RoPE")
	}
	if plan.FullFreqs[idx] != 1 || plan.FullFreqs[idx+1] == plain[idx+1] {
		t.Fatalf("suppressed pair got cos/sin=(%g,%g), plain=(%g,%g)", plan.FullFreqs[idx], plan.FullFreqs[idx+1], plain[idx], plain[idx+1])
	}
}
