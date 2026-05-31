package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGGUFParseConfigQwenNextHybridMetadata(t *testing.T) {
	g := &gguf.GGUF{Meta: map[string]any{
		"general.architecture":              "qwen35moe",
		"qwen35moe.embedding_length":        uint32(2048),
		"qwen35moe.block_count":             uint32(40),
		"qwen35moe.attention.head_count":    uint32(16),
		"qwen35moe.attention.head_count_kv": uint32(2),
		"qwen35moe.feed_forward_length":     uint32(512),
		"qwen35moe.context_length":          uint32(262144),
		"qwen35moe.attention.key_length":    uint32(256),
		"qwen35moe.attention.value_length":  uint32(256),
		"qwen35moe.full_attention_interval": uint32(4),
		"qwen35moe.ssm.conv_kernel":         uint32(4),
		"qwen35moe.ssm.group_count":         uint32(16),
		"qwen35moe.ssm.inner_size":          uint32(4096),
		"qwen35moe.ssm.state_size":          uint32(128),
		"qwen35moe.ssm.time_step_rank":      uint32(32),
	}}
	cfg, err := ggufParseConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsQwenNextHybridGGUF() || cfg.SSMInnerSize != 4096 || cfg.SSMConvKernel != 4 || cfg.AttentionKeyLength != 256 || cfg.FullAttentionInterval != 4 {
		t.Fatalf("unexpected hybrid cfg: %+v", cfg)
	}
	if err := cfg.ValidateQwenNextHybridMetadata(); err != nil {
		t.Fatal(err)
	}
}

func TestGGUFQwenNextHybridMetadataRequiresSSM(t *testing.T) {
	cfg := GGUFLlamaConfig{HiddenSize: 2048, AttentionKeyLength: 256, AttentionValueLength: 256, FullAttentionInterval: 4}
	if err := cfg.ValidateQwenNextHybridMetadata(); err == nil {
		t.Fatal("expected incomplete SSM metadata error")
	}
}
