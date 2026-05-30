package qwen3tts

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestValidateTensorShapes(t *testing.T) {
	cfg := ParsedConfig{TalkerHiddenSize: 1024, TalkerIntermediateSize: 3072, TalkerNumKeyValueHeads: 8, TalkerHeadDim: 64, TalkerTextHiddenSize: 2048, TalkerVocabSize: 3072, CPHiddenSize: 1024, CPIntermediateSize: 3072, CPNumKeyValueHeads: 8, CPHeadDim: 64}
	valid := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"talker.model.layers.0.self_attn.q_proj.weight":   {Shape: []int{1024, 1024}},
		"talker.model.layers.0.self_attn.k_proj.weight":   {Shape: []int{512, 1024}},
		"talker.model.layers.0.mlp.gate_proj.weight":      {Shape: []int{3072, 1024}},
		"code_predictor.layers.0.self_attn.o_proj.weight": {Shape: []int{1024, 1024}},
		"code_predictor.layers.0.self_attn.v_proj.weight": {Shape: []int{512, 1024}},
		"code_predictor.layers.0.mlp.down_proj.weight":    {Shape: []int{1024, 3072}},
		"talker.text_projection.weight":                   {Shape: []int{1024, 2048}},
		"talker.codec_head.weight":                        {Shape: []int{3072, 1024}},
		"model.codec_embedding.0.weight":                  {Shape: []int{2048, 1024}},
	})
	if !valid.Valid || len(valid.Issues) != 0 {
		t.Fatalf("valid=%+v", valid)
	}
	bad := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"talker.model.layers.0.self_attn.q_proj.weight":   {Shape: []int{1024, 512}},
		"talker.model.layers.0.self_attn.k_proj.weight":   {Shape: []int{1024, 1024}},
		"talker.model.layers.0.mlp.gate_proj.weight":      {Shape: []int{1024, 1024}},
		"code_predictor.layers.0.self_attn.v_proj.weight": {Shape: []int{1024, 1024}},
		"code_predictor.layers.0.mlp.down_proj.weight":    {Shape: []int{1024, 1024}},
		"model.codec_embedding.0.weight":                  {Shape: []int{2048, 512}},
	})
	if bad.Valid || len(bad.Issues) != 6 {
		t.Fatalf("bad=%+v", bad)
	}
}
