package diffusiongemma

import "testing"

func TestGGUFExpertDispatchStatsSubAndTotal(t *testing.T) {
	base := ggufExpertDispatchStats{FusedUsed: 2, LegacyGroupedUsed: 1, CPUFallback: 3, Q4PointerTable: 4, Q8PointerTable: 5, GPUAttemptNS: 100, CPUFallbackNS: 200}
	now := ggufExpertDispatchStats{FusedUsed: 7, LegacyGroupedUsed: 2, CPUFallback: 9, Q4PointerTable: 10, Q8PointerTable: 11, GPUAttemptNS: 600, CPUFallbackNS: 900}
	d := now.Sub(base)
	if d.FusedUsed != 5 || d.LegacyGroupedUsed != 1 || d.CPUFallback != 6 || d.Q4PointerTable != 6 || d.Q8PointerTable != 6 || d.GPUAttemptNS != 500 || d.CPUFallbackNS != 700 {
		t.Fatalf("unexpected diff: %+v", d)
	}
	if d.Total() != 12 {
		t.Fatalf("total=%d want 12", d.Total())
	}
}
