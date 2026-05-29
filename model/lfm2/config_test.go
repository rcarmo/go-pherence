package lfm2

import "testing"

func TestParseConfig(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"model_type":"lfm2_moe",
		"architectures":["Lfm2MoeForCausalLM"],
		"dtype":"bfloat16",
		"vocab_size":128000,
		"hidden_size":2048,
		"intermediate_size":7168,
		"num_hidden_layers":3,
		"num_attention_heads":32,
		"num_key_value_heads":8,
		"max_position_embeddings":128000,
		"layer_types":["conv","conv","full_attention"],
		"num_dense_layers":2,
		"num_experts":32,
		"num_experts_per_tok":4,
		"moe_intermediate_size":1792,
		"conv_L_cache":3,
		"norm_eps":0.00001,
		"norm_topk_prob":true,
		"routed_scaling_factor":1.0,
		"use_expert_bias":true,
		"tie_word_embeddings":true,
		"rope_parameters":{"rope_theta":5000000,"rope_type":"default"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HeadDim != 64 {
		t.Fatalf("head_dim=%d", cfg.HeadDim)
	}
	if cfg.ConvLayerCount() != 2 || cfg.FullAttentionLayerCount() != 1 {
		t.Fatalf("layer counts conv=%d attn=%d", cfg.ConvLayerCount(), cfg.FullAttentionLayerCount())
	}
}

func TestRejectBadLayerPattern(t *testing.T) {
	_, err := ParseConfig([]byte(`{"model_type":"lfm2_moe","hidden_size":2048,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_hidden_layers":2,"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`))
	if err == nil {
		t.Fatal("expected layer count error")
	}
}

func TestTensorCoverage(t *testing.T) {
	cov := InspectTensorNames([]string{"model.embed_tokens.weight", "model.layers.0.conv.weight", "model.layers.2.router.weight", "model.layers.2.experts.0.w1.weight", "lm_head.weight", "x"})
	if cov.Embedding != 1 || cov.Layers != 1 || cov.Router != 1 || cov.Experts != 1 || cov.LMHead != 1 || cov.Other != 1 {
		t.Fatalf("coverage=%+v", cov)
	}
}
