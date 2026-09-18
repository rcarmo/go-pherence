//go:build riscv64

package diffusiongemma

import (
	"os"
	"testing"
)

func TestTextWeightsCloseClearsQ80Cache(t *testing.T) {
	modelDir := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL")
	if modelDir == "" {
		t.Skip("set GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL to run model-backed Q80 close test")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	fp := weights.ForwardPlan()
	if len(fp.Layers) == 0 || fp.Layers[0].QProj == nil {
		t.Fatal("missing layer 0 q_proj binding")
	}
	if _, ok, err := k3Q80ForBinding(weights, fp.Layers[0].QProj); err != nil || !ok {
		t.Fatalf("failed to populate Q80 cache ok=%v err=%v", ok, err)
	}
	if entries, bytes := weights.Q80CacheEntries(), weights.Q80CacheBytes(); entries == 0 || bytes == 0 {
		t.Fatalf("expected Q80 cache entries before close, got entries=%d bytes=%d", entries, bytes)
	}
	if err := weights.Close(); err != nil {
		t.Fatal(err)
	}
	if entries, bytes := weights.Q80CacheEntries(), weights.Q80CacheBytes(); entries != 0 || bytes != 0 {
		t.Fatalf("expected Q80 cache to be cleared after close, got entries=%d bytes=%d", entries, bytes)
	}
}
