package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestMoEForwardGPUMalformedInputs(t *testing.T) {
	cfg := LlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 4, MoEIntermediate: 8}
	if got := moeForwardGPU(nil, nil, &LlamaLayer{}, cfg, nvidia.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("nil xDev returned %#v, want nil", got)
	}
	if got := moeForwardGPU(nil, nvidia.NewDevBuf(2), &LlamaLayer{}, cfg, nvidia.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("short xDev returned %#v, want nil", got)
	}
	badCfg := cfg
	badCfg.NumExperts = 0
	if got := moeForwardGPU(nil, nvidia.NewDevBuf(4), &LlamaLayer{}, badCfg, nvidia.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("zero experts returned %#v, want nil", got)
	}
}

func TestMoEForwardGPUSkipsIncompleteExpertWeights(t *testing.T) {
	cfg := LlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 2, MoEIntermediate: 8}
	layer := &LlamaLayer{
		// Router is nil, so equal probabilities select both experts. Expert 0 is
		// incomplete and must be skipped instead of panicking on missing up/down.
		ExpertGateW: make([]*mlx.QuantWeight, 2),
		ExpertUpW:   make([]*mlx.QuantWeight, 2),
		ExpertDownW: make([]*mlx.QuantWeight, 2),
	}
	layer.ExpertGateW[0] = &mlx.QuantWeight{}
	got := moeForwardGPU(nil, nvidia.NewDevBuf(4), layer, cfg, nil, 0, nil)
	if len(got) != cfg.HiddenSize {
		t.Fatalf("len=%d, want %d", len(got), cfg.HiddenSize)
	}
}

func TestMoEForwardGPUClampsActiveExperts(t *testing.T) {
	cfg := LlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 8, MoEIntermediate: 8}
	layer := &LlamaLayer{}
	got := moeForwardGPU(nil, nvidia.NewDevBuf(4), layer, cfg, nil, 0, nil)
	if len(got) != cfg.HiddenSize {
		t.Fatalf("len=%d, want %d", len(got), cfg.HiddenSize)
	}
}

func TestUploadExpertNativeToPoolRejectsMalformedSizes(t *testing.T) {
	layer := &LlamaLayer{
		ExpertGateW: []*mlx.QuantWeight{{}},
		ExpertUpW:   []*mlx.QuantWeight{{}},
		ExpertDownW: []*mlx.QuantWeight{{}},
	}
	pool := nvidia.NewExpertPool(1, nil)
	if got := uploadExpertNativeToPool(pool, layer, 0, 0, 0, 4); got != nil {
		t.Fatalf("zero moeInter returned %#v, want nil", got)
	}
	maxInt := int(^uint(0) >> 1)
	if got := uploadExpertNativeToPool(pool, layer, 0, 0, maxInt/2+1, 2); got != nil {
		t.Fatalf("overflowing size returned %#v, want nil", got)
	}
}
