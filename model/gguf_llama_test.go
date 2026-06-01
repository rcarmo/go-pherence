package model

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGGUFParseConfigQwenMoE(t *testing.T) {
	g := &gguf.GGUF{Meta: map[string]any{
		"general.architecture":                "qwen3moe",
		"qwen3moe.embedding_length":           uint32(4096),
		"qwen3moe.block_count":                uint32(48),
		"qwen3moe.attention.head_count":       uint32(32),
		"qwen3moe.attention.head_count_kv":    uint32(8),
		"qwen3moe.feed_forward_length":        uint32(12288),
		"qwen3moe.context_length":             uint32(32768),
		"qwen3moe.expert_count":               uint32(128),
		"qwen3moe.expert_used_count":          uint32(8),
		"qwen3moe.expert_feed_forward_length": uint32(1536),
		"qwen3moe.rope.freq_base":             float32(1000000),
	}}
	cfg, err := ggufParseConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Architecture != "qwen3moe" || cfg.HiddenSize != 4096 || cfg.NumExperts != 128 || cfg.NumExpertsPerTok != 8 || cfg.MoEHiddenSize != 1536 || cfg.MaxSeqLen != 32768 {
		t.Fatalf("unexpected cfg: %+v", cfg)
	}
}

func TestGGUFConfigEOS(t *testing.T) {
	cfg := GGUFLlamaConfig{VocabSize: 10, EOSTokenID: 7}
	if !cfg.IsEOS(7) || cfg.IsEOS(2) {
		t.Fatalf("explicit EOS mismatch")
	}
	cfg.EOSTokenID = 0
	if !cfg.IsEOS(2) || !cfg.IsEOS(9) || cfg.IsEOS(8) {
		t.Fatalf("fallback EOS mismatch")
	}
}

func TestGGUFParseConfigInfersVocabFromEmbedding(t *testing.T) {
	g := &gguf.GGUF{Meta: map[string]any{
		"general.architecture":          "llama",
		"llama.embedding_length":        uint32(4),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(1),
		"llama.attention.head_count_kv": uint32(1),
		"llama.feed_forward_length":     uint32(8),
		"llama.context_length":          uint32(16),
	}, Tensors: []gguf.TensorInfo{{Name: "token_embd.weight", Shape: []uint64{4, 99}}}}
	cfg, err := ggufParseConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VocabSize != 99 {
		t.Fatalf("vocab=%d", cfg.VocabSize)
	}
}

func TestGGUFMoEValidationRequiresCompleteMetadata(t *testing.T) {
	cfg := GGUFLlamaConfig{Architecture: "qwen3moe", NumExperts: 128, NumExpertsPerTok: 8}
	err := cfg.ValidateRuntimeSupported()
	if err == nil || !strings.Contains(err.Error(), "incomplete GGUF MoE metadata") {
		t.Fatalf("expected incomplete metadata error, got %v", err)
	}
	cfg.MoEHiddenSize = 1536
	if err := cfg.ValidateRuntimeSupported(); err != nil {
		t.Fatalf("complete MoE metadata rejected: %v", err)
	}
}
