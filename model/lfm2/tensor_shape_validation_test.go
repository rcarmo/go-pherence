package lfm2

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestValidateTensorShapes(t *testing.T) {
	cfg := Config{HiddenSize: 2048, NumExperts: 32, NumKeyValueHeads: 8, HeadDim: 64, MoEIntermediateSize: 1792, ConvLCache: 3}
	valid := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":                        {Shape: []int{128000, 2048}},
		"model.embedding_norm.weight":                      {Shape: []int{2048}},
		"model.layers.0.operator_norm.weight":              {Shape: []int{2048}},
		"model.layers.0.ffn_norm.weight":                   {Shape: []int{2048}},
		"model.layers.2.feed_forward.gate.weight":          {Shape: []int{32, 2048}},
		"model.layers.2.feed_forward.experts.gate_up_proj": {Shape: []int{32, 3584, 2048}},
		"model.layers.2.feed_forward.experts.down_proj":    {Shape: []int{32, 2048, 1792}},
		"model.layers.7.self_attn.q_proj.weight":           {Shape: []int{2048, 2048}},
		"model.layers.7.self_attn.k_proj.weight":           {Shape: []int{512, 2048}},
		"model.layers.7.self_attn.q_layernorm.weight":      {Shape: []int{64}},
		"model.layers.0.conv.in_proj.weight":               {Shape: []int{6144, 2048}},
		"model.layers.0.conv.out_proj.weight":              {Shape: []int{2048, 2048}},
		"model.layers.0.conv.conv.weight":                  {Shape: []int{2048, 1, 3}},
		"lm_head.weight":                                   {Shape: []int{128000, 2048}},
	})
	if !valid.Valid || len(valid.Issues) != 0 {
		t.Fatalf("valid=%+v", valid)
	}
	bad := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":                        {Shape: []int{127999, 2048}},
		"model.embedding_norm.weight":                      {Shape: []int{1024}},
		"model.layers.2.feed_forward.gate.weight":          {Shape: []int{16, 2048}},
		"model.layers.2.feed_forward.experts.gate_up_proj": {Shape: []int{32, 1792, 2048}},
		"model.layers.7.self_attn.q_proj.weight":           {Shape: []int{1024, 2048}},
		"model.layers.7.self_attn.k_proj.weight":           {Shape: []int{2048, 2048}},
		"model.layers.0.conv.conv.weight":                  {Shape: []int{2048, 1, 4}},
	})
	if bad.Valid || len(bad.Issues) != 7 {
		t.Fatalf("bad=%+v", bad)
	}
}
