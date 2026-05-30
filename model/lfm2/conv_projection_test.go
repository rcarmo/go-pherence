package lfm2

import (
	"path/filepath"
	"testing"
)

func TestConvProjectionLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewConvProjectionLayout(meta.Config, schedule)
	if err != nil {
		t.Fatal(err)
	}
	if layout.HiddenSize != 2048 || layout.ConvLCache != 3 || layout.ConvLayers != 21 || layout.HasBias {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.KernelFloats != 6144 || layout.BiasFloats != 0 || layout.FloatsPerLayer != 6144 || layout.TotalConvFloats != 129024 {
		t.Fatalf("floats=%+v", layout)
	}
}

func TestConvProjectionLayoutWithBias(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := meta.Config
	cfg.ConvBias = true
	layout, err := NewConvProjectionLayout(cfg, LayerSchedule{})
	if err != nil {
		t.Fatal(err)
	}
	if !layout.HasBias || layout.BiasFloats != cfg.HiddenSize || layout.FloatsPerLayer != 8192 {
		t.Fatalf("layout=%+v", layout)
	}
}

func TestConvProjectionLayoutRejectsMalformed(t *testing.T) {
	bad := ConvProjectionLayout{HiddenSize: 2048, ConvLCache: 3, ConvLayers: 21, KernelFloats: 1, FloatsPerLayer: 6144, TotalConvFloats: 129024}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected kernel float error")
	}
	bad = ConvProjectionLayout{HiddenSize: 2048, ConvLCache: 3, ConvLayers: 21, HasBias: true, KernelFloats: 6144, BiasFloats: 0, FloatsPerLayer: 6144, TotalConvFloats: 129024}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected bias float error")
	}
	bad = ConvProjectionLayout{HiddenSize: 2048, ConvLCache: 3, ConvLayers: 21, KernelFloats: 6144, FloatsPerLayer: 1, TotalConvFloats: 129024}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected floats/layer error")
	}
}
