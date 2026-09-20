package diffusiongemma

import (
	"os"
	"testing"
)

func TestOpenCPUExpertIndexAdmission(t *testing.T) {
	if _, _, err := OpenCPUExpertIndex("", Shape{}, nil); err == nil {
		t.Fatal("nil weights admitted")
	}
	fused := &TextWeights{}
	idx, owner, err := OpenCPUExpertIndex("missing", Shape{}, fused)
	if err != nil || idx != nil || owner != nil {
		t.Fatalf("fused checkpoint changed: %v", err)
	}
	indexed := &TextWeights{IndexedExperts: true}
	if _, owner, err = OpenCPUExpertIndex("missing", Shape{TextLayers: 1, NumExperts: 1}, indexed); err == nil || owner != nil {
		t.Fatal("missing indexed checkpoint admitted")
	}
}
func TestLocalFP8CPUExpertIndex(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_FP8_MODEL")
	if dir == "" {
		t.Skip("set local FP8 checkpoint")
	}
	m, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeights(dir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	idx, owner, err := OpenCPUExpertIndex(dir, m.Shape, weights)
	if err != nil {
		t.Fatal(err)
	}
	if owner == nil || idx == nil {
		t.Fatal("indexed FP8 checkpoint did not produce index")
	}
	defer owner.Close()
	if idx.NumLayers != m.Shape.TextLayers || idx.NumExperts != m.Shape.NumExperts || idx.HiddenSize != m.Shape.TextHiddenSize || idx.Intermediate != m.Shape.MoEIntermediateSize {
		t.Fatalf("unexpected index %+v", idx)
	}
}
