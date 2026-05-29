package lfm2

import (
	"path/filepath"
	"testing"
)

func TestLoadReferenceMetadata(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := meta.Config
	if cfg.ModelType != ModelType || cfg.HiddenSize != 2048 || cfg.NumHiddenLayers != 24 {
		t.Fatalf("config=%+v", cfg)
	}
	if cfg.ConvLayerCount() != 21 || cfg.FullAttentionLayerCount() != 3 {
		t.Fatalf("layer counts conv=%d attn=%d", cfg.ConvLayerCount(), cfg.FullAttentionLayerCount())
	}
	if cfg.NumExperts != 32 || cfg.NumExpertsPerTok != 4 || cfg.ConvLCache != 3 {
		t.Fatalf("moe/conv config=%+v", cfg)
	}
}
