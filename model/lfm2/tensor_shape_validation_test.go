package lfm2

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestValidateTensorShapes(t *testing.T) {
	cfg := Config{HiddenSize: 2048, NumExperts: 32, MoEIntermediateSize: 1792}
	valid := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":    {Shape: []int{128000, 2048}},
		"model.layers.2.router.weight": {Shape: []int{32, 2048}},
		"lm_head.weight":               {Shape: []int{128000, 2048}},
	})
	if !valid.Valid || len(valid.Issues) != 0 {
		t.Fatalf("valid=%+v", valid)
	}
	bad := ValidateTensorShapes(cfg, map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":    {Shape: []int{128000, 1024}},
		"model.layers.2.router.weight": {Shape: []int{16, 2048}},
	})
	if bad.Valid || len(bad.Issues) != 2 {
		t.Fatalf("bad=%+v", bad)
	}
}
