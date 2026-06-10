package model

import (
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
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
