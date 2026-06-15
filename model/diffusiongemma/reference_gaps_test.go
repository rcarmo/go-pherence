package diffusiongemma

import "testing"

func TestMissingReferenceGapsCopySafe(t *testing.T) {
	gaps := MissingReferenceGaps()
	if len(gaps) != 2 || gaps[0] != "broader reference parity fixtures" || gaps[1] != "full image-sequence vision reference fixtures" {
		t.Fatalf("unexpected gaps: %v", gaps)
	}
	gaps[0] = "mutated"
	again := MissingReferenceGaps()
	if again[0] == "mutated" {
		t.Fatalf("MissingReferenceGaps returned shared mutable storage: %v", again)
	}
}
