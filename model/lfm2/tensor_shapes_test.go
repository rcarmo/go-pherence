package lfm2

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestInspectTensorShapes(t *testing.T) {
	s := InspectTensorShapes(map[string]safetensors.TensorInfo{
		"model.embed_tokens.weight":       {DType: "BF16", Shape: []int{128000, 2048}},
		"model.layers.2.router.weight":    {DType: "F32", Shape: []int{32, 2048}},
		"model.layers.2.experts.0.weight": {DType: "BF16", Shape: []int{1792, 2048}},
	})
	if s.Total != 3 || s.ByGroup["embedding"] != 1 || s.ByGroup["router"] != 1 || s.ByGroup["experts"] != 1 || s.DTypes["BF16"] != 2 {
		t.Fatalf("summary=%+v", s)
	}
	if s.Examples["embedding"].Shape[1] != 2048 {
		t.Fatalf("examples=%+v", s.Examples)
	}
}
