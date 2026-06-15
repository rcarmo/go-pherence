package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestTiedMLXEmbeddingCanBackCompactLMHead(t *testing.T) {
	emb := &mlx.QuantWeight{InDim: 4, OutDim: 8, GroupSize: 4, Weight: []uint32{1, 2}, Scales: []float32{1}, Biases: []float32{0}}
	m := &LlamaModel{EmbedTokens: tensor.FromFloat32(make([]float32, 32), []int{8, 4})}
	m.LMHead = m.EmbedTokens
	m.LMHeadMLX = emb
	if m.LMHeadMLX != emb {
		t.Fatal("tied MLX embedding was not retained for compact LM head")
	}
}

func TestGPUPackedWeightDimGuards(t *testing.T) {
	if !gpuMLXWeightDims(&nvidia.GPUMLXWeight{OutDim: 4, InDim: 8}, 4, 8) {
		t.Fatal("valid GPU MLX dims rejected")
	}
	if gpuMLXWeightDims(&nvidia.GPUMLXWeight{OutDim: 4, InDim: 8}, 8, 4) {
		t.Fatal("mismatched GPU MLX dims accepted")
	}
	if !gpuQ4WeightDims(&nvidia.GPUQuantWeight{OutDim: 4, InDim: 8}, 4, 8) {
		t.Fatal("valid GPU Q4 dims rejected")
	}
	if gpuQ4WeightDims(&nvidia.GPUQuantWeight{OutDim: 4, InDim: 8}, 4, 4) {
		t.Fatal("mismatched GPU Q4 dims accepted")
	}
	if !cpuQ4WeightDims(&QuantWeight{OutDim: 4, InDim: 8}, 4, 8) {
		t.Fatal("valid CPU Q4 dims rejected")
	}
	if cpuQ4WeightDims(&QuantWeight{OutDim: 4, InDim: 8}, 8, 8) {
		t.Fatal("mismatched CPU Q4 dims accepted")
	}
}

func TestLayerKVHeadsForGPUKVBuffersUsesGemmaFullAttentionHeads(t *testing.T) {
	cfg := LlamaConfig{NumKVHeads: 16, NumGlobalKVHeads: 4, HeadDim: 256, GlobalHeadDim: 512, LayerTypes: []string{"sliding_attention", "full_attention"}}
	if got := gemmacfg.LayerKVHeads(cfg, 0) * 256; got != 4096 {
		t.Fatalf("sliding GPU KV dim=%d want 4096", got)
	}
	if got := gemmacfg.LayerKVHeads(cfg, 1) * 512; got != 2048 {
		t.Fatalf("full GPU KV dim=%d want 2048", got)
	}
}

func TestGemma4LayerKVDimUsesGlobalKVHeadsForFullAttention(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 16, NumGlobalKVHeads: 4, HeadDim: 256, GlobalHeadDim: 512, LayerTypes: []string{"sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true, HeadDimLocal: 256}, {HasKV: true, HeadDimLocal: 512}},
	}
	if got, err := m.LayerKVDim(0); err != nil || got != 4096 {
		t.Fatalf("sliding LayerKVDim=%d err=%v want 4096", got, err)
	}
	if got, err := m.LayerKVDim(1); err != nil || got != 2048 {
		t.Fatalf("full LayerKVDim=%d err=%v want 2048", got, err)
	}
}
