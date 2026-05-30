package lfm2

import (
	"path/filepath"
	"testing"
)

func TestAttentionKVLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewAttentionKVLayout(meta.Config, schedule)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Layers != 3 || layout.KVHeads != 8 || layout.HeadDim != 64 || layout.FloatsPerToken != 3072 {
		t.Fatalf("layout=%+v", layout)
	}
	want := []int{7, 15, 23}
	for i := range want {
		if layout.LayerIndices[i] != want[i] {
			t.Fatalf("layer indices=%v", layout.LayerIndices)
		}
	}
	bytes, err := layout.Bytes(128, 2)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 786432 {
		t.Fatalf("bytes=%d", bytes)
	}
}

func TestAttentionKVLayoutAllowsNoAttentionLayers(t *testing.T) {
	layout := AttentionKVLayout{Layers: 0, KVHeads: 8, HeadDim: 64, FloatsPerToken: 0}
	if err := layout.Validate(); err != nil {
		t.Fatal(err)
	}
	bytes, err := layout.Bytes(128, 2)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 0 {
		t.Fatalf("bytes=%d", bytes)
	}
}

func TestAttentionKVLayoutRejectsMalformed(t *testing.T) {
	bad := AttentionKVLayout{Layers: 1, KVHeads: 8, HeadDim: 64, FloatsPerToken: 1, LayerIndices: []int{7}}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected floats/token error")
	}
	bad = AttentionKVLayout{Layers: 2, KVHeads: 8, HeadDim: 64, FloatsPerToken: 2048, LayerIndices: []int{15, 7}}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected layer order error")
	}
	if _, err := bad.Bytes(-1, 2); err == nil {
		t.Fatal("expected max sequence error")
	}
}
