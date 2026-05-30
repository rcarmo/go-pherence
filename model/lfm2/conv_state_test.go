package lfm2

import (
	"path/filepath"
	"testing"
)

func TestConvStateLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewConvStateLayout(meta.Config, schedule)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Layers != 21 || layout.HiddenSize != 2048 || layout.LCache != 3 || layout.FloatsPerLayer != 6144 || layout.TotalFloats != 129024 {
		t.Fatalf("layout=%+v", layout)
	}
	bytes, err := layout.Bytes(2)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 258048 {
		t.Fatalf("bytes=%d", bytes)
	}
}

func TestConvStateLayoutRejectsMalformed(t *testing.T) {
	bad := ConvStateLayout{Layers: 1, HiddenSize: 2048, LCache: 3, FloatsPerLayer: 1, TotalFloats: 1, LayerIndices: []int{0}}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected floats/layer error")
	}
	bad = ConvStateLayout{Layers: 2, HiddenSize: 2048, LCache: 3, FloatsPerLayer: 6144, TotalFloats: 12288, LayerIndices: []int{2, 1}}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected layer order error")
	}
	if _, err := bad.Bytes(0); err == nil {
		t.Fatal("expected bytes/float error")
	}
}
