package lfm2

import (
	"path/filepath"
	"testing"
)

func TestEmbeddingLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewEmbeddingLayout(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if layout.VocabSize != 128000 || layout.HiddenSize != 2048 || !layout.TieWordEmbeddings || !layout.OutputSharesInput {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.EmbeddingFloats != 262144000 || layout.LMHeadFloats != 0 || layout.TotalUntiedFloats != 262144000 {
		t.Fatalf("layout floats=%+v", layout)
	}
	bytes, err := layout.Bytes(2)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 524288000 {
		t.Fatalf("bytes=%d", bytes)
	}
}

func TestEmbeddingLayoutUntied(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := meta.Config
	cfg.VocabSize = 100
	cfg.TieWordEmbeddings = false
	layout, err := NewEmbeddingLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if layout.EmbeddingFloats != 100*cfg.HiddenSize || layout.LMHeadFloats != 100*cfg.HiddenSize || layout.TotalUntiedFloats != 2*100*cfg.HiddenSize || layout.OutputSharesInput {
		t.Fatalf("layout=%+v", layout)
	}
}

func TestEmbeddingLayoutRejectsMalformed(t *testing.T) {
	bad := EmbeddingLayout{VocabSize: 100, HiddenSize: 64, TieWordEmbeddings: true, EmbeddingFloats: 1, OutputSharesInput: true, TotalUntiedFloats: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected embedding float error")
	}
	bad = EmbeddingLayout{VocabSize: 100, HiddenSize: 64, TieWordEmbeddings: false, EmbeddingFloats: 6400, LMHeadFloats: 0, OutputSharesInput: false, TotalUntiedFloats: 6400}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected lm_head float error")
	}
	bad = EmbeddingLayout{VocabSize: 100, HiddenSize: 64, TieWordEmbeddings: true, EmbeddingFloats: 6400, OutputSharesInput: false, TotalUntiedFloats: 6400}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected sharing flag error")
	}
	if _, err := (EmbeddingLayout{VocabSize: 100, HiddenSize: 64, TieWordEmbeddings: true, EmbeddingFloats: 6400, OutputSharesInput: true, TotalUntiedFloats: 6400}).Bytes(0); err == nil {
		t.Fatal("expected byte sizing error")
	}
}
