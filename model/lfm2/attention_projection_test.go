package lfm2

import (
	"path/filepath"
	"testing"
)

func TestAttentionProjectionLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewAttentionProjectionLayout(meta.Config, schedule)
	if err != nil {
		t.Fatal(err)
	}
	if layout.HiddenSize != 2048 || layout.Heads != 32 || layout.KVHeads != 8 || layout.HeadDim != 64 || layout.QueriesPerKV != 4 {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.QLayerFloats != 4194304 || layout.KLayerFloats != 1048576 || layout.VLayerFloats != 1048576 || layout.OLayerFloats != 4194304 {
		t.Fatalf("projection floats=%+v", layout)
	}
	if layout.FullAttentionLayers != 3 || layout.TotalFloatsPerLayer != 10485760 || layout.TotalAttentionFloats != 31457280 {
		t.Fatalf("totals=%+v", layout)
	}
}

func TestAttentionProjectionLayoutRejectsMalformed(t *testing.T) {
	bad := AttentionProjectionLayout{HiddenSize: 2048, Heads: 32, KVHeads: 8, HeadDim: 64, QueriesPerKV: 1, QLayerFloats: 4194304, KLayerFloats: 1048576, VLayerFloats: 1048576, OLayerFloats: 4194304, TotalFloatsPerLayer: 10485760, FullAttentionLayers: 3, TotalAttentionFloats: 31457280}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected queries/kv error")
	}
	bad = AttentionProjectionLayout{HiddenSize: 2048, Heads: 32, KVHeads: 8, HeadDim: 64, QueriesPerKV: 4, QLayerFloats: 1, KLayerFloats: 1048576, VLayerFloats: 1048576, OLayerFloats: 4194304, TotalFloatsPerLayer: 10485760, FullAttentionLayers: 3, TotalAttentionFloats: 31457280}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected projection size error")
	}
	bad = AttentionProjectionLayout{HiddenSize: 2048, Heads: 32, KVHeads: 8, HeadDim: 64, QueriesPerKV: 4, QLayerFloats: 4194304, KLayerFloats: 1048576, VLayerFloats: 1048576, OLayerFloats: 4194304, TotalFloatsPerLayer: 10485760, FullAttentionLayers: 3, TotalAttentionFloats: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected total size error")
	}
}
