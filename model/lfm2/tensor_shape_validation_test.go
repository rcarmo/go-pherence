package lfm2

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestValidateTensorShapes(t *testing.T) {
	cfg := Config{HiddenSize: 2048, NumExperts: 32, NumKeyValueHeads: 8, HeadDim: 64, MoEIntermediateSize: 1792, ConvLCache: 3}
	valid := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":       {Shape: []int{128000, 2048}},
		"model.layers.2.router.weight":    {Shape: []int{32, 2048}},
		"model.layers.7.self_attn.q_proj": {Shape: []int{2048, 2048}},
		"model.layers.7.self_attn.k_proj": {Shape: []int{512, 2048}},
		"model.layers.0.conv.weight":      {Shape: []int{2048, 3}},
		"lm_head.weight":                  {Shape: []int{128000, 2048}},
	})
	if !valid.Valid || len(valid.Issues) != 0 {
		t.Fatalf("valid=%+v", valid)
	}
	bad := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":       {Shape: []int{128000, 1024}},
		"model.layers.2.router.weight":    {Shape: []int{16, 2048}},
		"model.layers.7.self_attn.q_proj": {Shape: []int{1024, 2048}},
		"model.layers.7.self_attn.k_proj": {Shape: []int{2048, 2048}},
		"model.layers.0.conv.weight":      {Shape: []int{2048, 4}},
	})
	if bad.Valid || len(bad.Issues) != 5 {
		t.Fatalf("bad=%+v", bad)
	}
}
