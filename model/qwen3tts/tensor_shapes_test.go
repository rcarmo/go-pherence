package qwen3tts

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestInspectTensorShapes(t *testing.T) {
	s := InspectTensorShapes(map[string]safetensors.TensorInfo{
		"talker.model.layers.0.self_attn.q_proj.weight": {DType: "BF16", Shape: []int{1024, 1024}},
		"model.codec_embedding.0.weight":                {DType: "F32", Shape: []int{2048, 1024}},
	})
	if s.Total != 2 || s.ByGroup["talker"] != 1 || s.ByGroup["code_predictor"] != 1 || s.DTypes["BF16"] != 1 || s.DTypes["F32"] != 1 {
		t.Fatalf("summary=%+v", s)
	}
	if s.Examples["talker"].Shape[0] != 1024 {
		t.Fatalf("examples=%+v", s.Examples)
	}
}
