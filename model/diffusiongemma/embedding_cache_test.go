package diffusiongemma

import "testing"

func TestCachedScaledEmbeddingRowRejectsNilWeights(t *testing.T) {
	if _, err := cachedScaledEmbeddingRow(nil, "x", 1, 4); err == nil {
		t.Fatal("expected nil weights error")
	}
}
