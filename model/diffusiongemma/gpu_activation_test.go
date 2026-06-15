package diffusiongemma

import "testing"

func TestF32GELUExactMulScratchReuseAndFree(t *testing.T) {
	freeF32GELUExactMulScratch()
	gate, up := ensureF32GELUExactMulScratch(4)
	if len(gate) != 4 || len(up) != 4 {
		t.Fatalf("scratch lens gate=%d up=%d want 4", len(gate), len(up))
	}
	gate[0], up[0] = 1, 2
	gate2, up2 := ensureF32GELUExactMulScratch(2)
	if len(gate2) != 2 || len(up2) != 2 || gate2[0] != 1 || up2[0] != 2 {
		t.Fatalf("scratch was not reused: gate=%v up=%v", gate2, up2)
	}
	freeF32GELUExactMulScratch()
	if f32GELUExactMulScratch.gate != nil || f32GELUExactMulScratch.up != nil {
		t.Fatalf("scratch not freed")
	}
}
