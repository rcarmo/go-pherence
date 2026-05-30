package lfm2

import (
	"path/filepath"
	"testing"
)

func TestContextLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewContextLayout(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if layout.VocabSize != 128000 || layout.MaxPositionEmbeddings != 128000 || !layout.TieWordEmbeddings || layout.RoPETheta != 5000000 {
		t.Fatalf("layout=%+v", layout)
	}
	if err := layout.ValidateSequence([]uint32{0, 127999}); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateToken(128000); err == nil {
		t.Fatal("expected token range error")
	}
}

func TestContextLayoutRejectsMalformed(t *testing.T) {
	bad := ContextLayout{VocabSize: 0, MaxPositionEmbeddings: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected vocab error")
	}
	layout := ContextLayout{VocabSize: 10, MaxPositionEmbeddings: 2}
	if err := layout.ValidateSequence(nil); err == nil {
		t.Fatal("expected empty sequence error")
	}
	if err := layout.ValidateSequence([]uint32{1, 2, 3}); err == nil {
		t.Fatal("expected context length error")
	}
}
