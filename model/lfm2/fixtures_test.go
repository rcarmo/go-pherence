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
	cov := meta.Coverage()
	if !cov.ConfigMetadata || !cov.RuntimePlan || cov.TensorCoverage || cov.CompleteRuntimeTrace {
		t.Fatalf("coverage=%+v", cov)
	}
}

func TestReferenceMetadataCoverageWithTensorReadiness(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	meta.Tensors = InspectTensorNames([]string{
		"model.embed_tokens.weight",
		"model.layers.0.conv.weight",
		"model.layers.2.router.weight",
		"model.layers.2.experts.0.w1.weight",
	})
	cov := meta.Coverage()
	if !cov.ConfigMetadata || !cov.RuntimePlan || !cov.TensorCoverage || !cov.TensorReadiness || cov.CompleteRuntimeTrace {
		t.Fatalf("coverage=%+v", cov)
	}
	wantMissing := map[string]bool{"tokenization_fixture": true, "first_token_logits": true, "conv_layer_reference": true, "attention_reference": true, "router_topk_reference": true, "expert_output_fixture": true}
	for _, name := range cov.Missing {
		delete(wantMissing, name)
	}
	if len(wantMissing) != 0 {
		t.Fatalf("missing did not include references: coverage=%+v remaining=%v", cov, wantMissing)
	}
}
