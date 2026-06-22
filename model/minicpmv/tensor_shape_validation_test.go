package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestValidateTensorShapesValid(t *testing.T) {
	summary := config.MiniCPMVSummary{HiddenSize: 4, KVHeads: 1, HeadDim: 2, IntermediateSize: 8, VisionHiddenSize: 3, NumQuery: 2, PatchSize: 14}
	infos := map[string]safetensors.TensorInfo{
		"llm.model.embed_tokens.weight":              {Shape: []int{100, 4}},
		"llm.model.layers.0.self_attn.q_proj.weight": {Shape: []int{4, 4}},
		"llm.model.layers.0.self_attn.k_proj.weight": {Shape: []int{2, 4}},
		"llm.model.layers.0.mlp.gate_proj.weight":    {Shape: []int{8, 4}},
		"llm.lm_head.weight":                         {Shape: []int{100, 4}},
		"resampler.query.weight":                     {Shape: []int{2, 4}},
		"resampler.kv_proj.weight":                   {Shape: []int{4, 3}},
		"vpm.embeddings.patch_embedding.weight":      {Shape: []int{3, 3, 14, 14}},
	}
	if v := ValidateTensorShapes(summary, infos); !v.Valid || len(v.Issues) != 0 {
		t.Fatalf("expected valid tensor shapes: %+v", v)
	}
}

func TestValidateTensorShapesAudio(t *testing.T) {
	summary := config.MiniCPMVSummary{AudioHiddenSize: 4, AudioFeatureSize: 80}
	valid := ValidateTensorShapes(summary, map[string]safetensors.TensorInfo{
		"audio_encoder.conv1.weight":                     {Shape: []int{4, 80, 3}},
		"audio_encoder.layers.0.self_attn.q_proj.weight": {Shape: []int{4, 4}},
	})
	if !valid.Valid {
		t.Fatalf("expected valid audio shapes: %+v", valid)
	}
	bad := ValidateTensorShapes(summary, map[string]safetensors.TensorInfo{
		"audio_encoder.conv1.weight":                     {Shape: []int{8, 80, 3}},
		"audio_encoder.layers.0.self_attn.q_proj.weight": {Shape: []int{8, 8}},
	})
	if bad.Valid || len(bad.Issues) < 2 {
		t.Fatalf("expected audio shape issues: %+v", bad)
	}
}

func TestValidateTensorShapesRejectsMismatches(t *testing.T) {
	summary := config.MiniCPMVSummary{HiddenSize: 4, KVHeads: 1, HeadDim: 2, IntermediateSize: 8, VisionHiddenSize: 3, NumQuery: 2, PatchSize: 14}
	infos := map[string]safetensors.TensorInfo{
		"llm.model.embed_tokens.weight":              {Shape: []int{100, 5}},
		"llm.model.layers.0.self_attn.k_proj.weight": {Shape: []int{4, 4}},
		"resampler.query.weight":                     {Shape: []int{3, 4}},
		"vpm.embeddings.patch_embedding.weight":      {Shape: []int{3, 3, 16, 16}},
	}
	v := ValidateTensorShapes(summary, infos)
	if v.Valid || len(v.Issues) < 4 {
		t.Fatalf("expected shape errors, got %+v", v)
	}
}
