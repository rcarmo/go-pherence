package lfm2

import (
	"path/filepath"
	"testing"
)

func TestNormLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewNormLayout(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if layout.HiddenSize != 2048 || layout.Layers != 24 || layout.Epsilon != 1e-5 || layout.TotalNormVectors != 48 {
		t.Fatalf("layout=%+v", layout)
	}
	scratch, err := layout.ScratchFloats(3)
	if err != nil {
		t.Fatal(err)
	}
	if scratch != 6144 {
		t.Fatalf("scratch=%d", scratch)
	}
}

func TestNormLayoutRejectsMalformed(t *testing.T) {
	bad := NormLayout{HiddenSize: 2048, Layers: 24, Epsilon: 1e-5, NormsPerLayer: 2, TotalNormVectors: 1, FloatsPerVector: 2048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected vector count error")
	}
	bad = NormLayout{HiddenSize: 2048, Layers: 24, Epsilon: 0, NormsPerLayer: 2, TotalNormVectors: 48, FloatsPerVector: 2048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected epsilon error")
	}
	if _, err := (NormLayout{HiddenSize: 2048, Layers: 1, Epsilon: 1e-5, NormsPerLayer: 2, TotalNormVectors: 2, FloatsPerVector: 2048}).ScratchFloats(-1); err == nil {
		t.Fatal("expected token count error")
	}
}
