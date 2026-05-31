package lfm2

import (
	"path/filepath"
	"testing"
)

func TestRuntimeRequestPlan(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(meta.Config, RuntimeRequest{Tokens: []uint32{1, 2, 3}, MaxNewTokens: 5, BytesPerFloat: 2})
	if err != nil {
		t.Fatal(err)
	}
	if plan.PromptTokens != 3 || plan.MaxSequence != 8 || plan.KVBytes != 49152 || plan.ConvStateBytes != 258048 || plan.RouterScratch != 320 || plan.EmbeddingBytes != 524288000 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestRuntimeRequestPlanRejectsMalformed(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRuntimeRequestPlan(meta.Config, RuntimeRequest{Tokens: nil, MaxNewTokens: 1, BytesPerFloat: 2}); err == nil {
		t.Fatal("expected empty token error")
	}
	if _, err := NewRuntimeRequestPlan(meta.Config, RuntimeRequest{Tokens: []uint32{uint32(meta.Config.VocabSize)}, MaxNewTokens: 1, BytesPerFloat: 2}); err == nil {
		t.Fatal("expected token range error")
	}
	if _, err := NewRuntimeRequestPlan(meta.Config, RuntimeRequest{Tokens: []uint32{1}, MaxNewTokens: meta.Config.MaxPositionEmbeddings, BytesPerFloat: 2}); err == nil {
		t.Fatal("expected context overflow")
	}
	plan, err := NewRuntimeRequestPlan(meta.Config, RuntimeRequest{Tokens: []uint32{1}, MaxNewTokens: 1, BytesPerFloat: 2})
	if err != nil {
		t.Fatal(err)
	}
	plan.MaxSequence++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected max sequence mismatch")
	}
}
