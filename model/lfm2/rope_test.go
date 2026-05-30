package lfm2

import (
	"path/filepath"
	"testing"
)

func TestRoPELayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewRoPELayout(meta.Config, schedule)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Theta != 5000000 || layout.HeadDim != 64 || layout.MaxPositionEmbeddings != 128000 || layout.FullAttentionLayers != 3 {
		t.Fatalf("layout=%+v", layout)
	}
	if err := layout.ValidatePosition(127999); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidatePosition(128000); err == nil {
		t.Fatal("expected position range error")
	}
}

func TestRoPELayoutRejectsMalformed(t *testing.T) {
	bad := RoPELayout{Theta: -1, HeadDim: 64, MaxPositionEmbeddings: 128000}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected theta error")
	}
	bad = RoPELayout{Theta: 1, HeadDim: 0, MaxPositionEmbeddings: 128000}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected head dim error")
	}
	if err := (RoPELayout{Theta: 1, HeadDim: 64, MaxPositionEmbeddings: 2}).ValidatePosition(-1); err == nil {
		t.Fatal("expected negative position error")
	}
}
