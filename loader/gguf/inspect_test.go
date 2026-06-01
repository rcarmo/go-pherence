package gguf

import "testing"

func TestInspectOpenDetectsQ4KMoEREAP(t *testing.T) {
	g := &GGUF{
		Meta: map[string]any{
			"general.architecture":       "qwen3moe",
			"general.name":               "Qwen3.6-REAP",
			"qwen3moe.expert_count":      uint32(128),
			"qwen3moe.expert_used_count": uint32(8),
			"qwen3moe.reap.prune_ratio":  float32(0.2),
		},
		Tensors: []TensorInfo{
			{Name: "blk.0.attn_q.weight", QType: QuantQ4_K},
			{Name: "blk.0.ffn_gate_exps.weight", QType: QuantQ4_K},
			{Name: "blk.0.ffn_down_exps.weight", QType: QuantQ6_K},
		},
	}
	in := InspectOpen("model.gguf", g)
	if in.Architecture != "qwen3moe" || in.Name != "Qwen3.6-REAP" {
		t.Fatalf("bad identity: %+v", in)
	}
	if !in.HasQ4K || !in.HasMoE || !in.HasREAPMetadata || !in.TurboQuantReady || !in.PureGoSIMDReady {
		t.Fatalf("bad readiness flags: %+v", in)
	}
	if in.RuntimeSupported || len(in.MissingRuntimeTensors) == 0 {
		t.Fatalf("synthetic index should not claim runtime support: %+v", in)
	}
	if in.Experts != 128 || in.ExpertsPerToken != 8 || in.QuantCounts["Q4_K"] != 2 || in.QuantCounts["Q6_K"] != 1 {
		t.Fatalf("bad counts: %+v", in)
	}
}

func TestInspectOpenRuntimeSupportedForExpectedMoETensors(t *testing.T) {
	g := &GGUF{Meta: map[string]any{"general.architecture": "llama", "llama.expert_count": uint32(2), "llama.expert_used_count": uint32(1)}, Tensors: []TensorInfo{
		{Name: "token_embd.weight", QType: QuantQ4_K}, {Name: "output_norm.weight", QType: QuantF32}, {Name: "output.weight", QType: QuantQ4_K},
		{Name: "blk.0.attn_q.weight", QType: QuantQ4_K}, {Name: "blk.0.attn_k.weight", QType: QuantQ4_K}, {Name: "blk.0.attn_v.weight", QType: QuantQ4_K}, {Name: "blk.0.attn_output.weight", QType: QuantQ4_K},
		{Name: "blk.0.attn_norm.weight", QType: QuantF32}, {Name: "blk.0.ffn_norm.weight", QType: QuantF32},
		{Name: "blk.0.ffn_gate_inp.weight", QType: QuantF32}, {Name: "blk.0.ffn_gate_exps.weight", QType: QuantQ4_K}, {Name: "blk.0.ffn_up_exps.weight", QType: QuantQ4_K}, {Name: "blk.0.ffn_down_exps.weight", QType: QuantQ4_K},
	}}
	in := InspectOpen("model.gguf", g)
	if !in.RuntimeSupported || len(in.MissingRuntimeTensors) != 0 {
		t.Fatalf("expected runtime support: %+v", in)
	}
}

func TestInspectOpenQwenNextRuntimeSupportedWhenTensorsPresent(t *testing.T) {
	g := &GGUF{Meta: map[string]any{"general.architecture": "qwen35moe", "qwen35moe.block_count": uint32(40), "qwen35moe.embedding_length": uint32(2048), "qwen35moe.attention.head_count": uint32(8), "qwen35moe.attention.head_count_kv": uint32(2), "qwen35moe.attention.key_length": uint32(256), "qwen35moe.full_attention_interval": uint32(4), "qwen35moe.ssm.inner_size": uint32(4096), "qwen35moe.expert_count": uint32(205), "qwen35moe.expert_used_count": uint32(8)}, Tensors: []TensorInfo{
		{Name: "token_embd.weight", QType: QuantQ4_K}, {Name: "output_norm.weight", QType: QuantF32}, {Name: "output.weight", QType: QuantQ4_K},
		{Name: "blk.0.attn_qkv.weight", QType: QuantQ4_K}, {Name: "blk.0.attn_gate.weight", QType: QuantQ4_K}, {Name: "blk.0.post_attention_norm.weight", QType: QuantF32},
		{Name: "blk.0.ssm_conv1d.weight", QType: QuantF32}, {Name: "blk.0.ssm_a", QType: QuantF32}, {Name: "blk.0.ssm_dt.bias", QType: QuantF32}, {Name: "blk.0.ssm_norm.weight", QType: QuantF32}, {Name: "blk.0.ssm_alpha.weight", QType: QuantQ4_K}, {Name: "blk.0.ssm_beta.weight", QType: QuantQ4_K}, {Name: "blk.0.ssm_out.weight", QType: QuantQ4_K},
		{Name: "blk.0.ffn_gate_inp.weight", QType: QuantF32}, {Name: "blk.0.ffn_gate_exps.weight", QType: QuantQ4_K}, {Name: "blk.0.ffn_up_exps.weight", QType: QuantQ4_K}, {Name: "blk.0.ffn_down_exps.weight", QType: QuantQ6_K},
	}}
	in := InspectOpen("qwen.gguf", g)
	if !in.RuntimeSupported || len(in.MissingRuntimeTensors) != 0 {
		t.Fatalf("expected runtime support: %+v", in)
	}
	if in.Layers != 40 || in.KVHeads != 2 || in.HeadDim != 256 || in.KVDim != 512 || in.CompressedKVLayers != 10 {
		t.Fatalf("bad QwenNext KV plan dims: %+v", in)
	}
}

func TestInspectOpenWarnsMissingMoEMetadata(t *testing.T) {
	g := &GGUF{Meta: map[string]any{"general.architecture": "qwen3moe"}, Tensors: []TensorInfo{{Name: "blk.0.ffn_gate_exps.weight", QType: QuantQ4_K}}}
	in := InspectOpen("model.gguf", g)
	if !in.HasMoE || len(in.ReadinessWarnings) == 0 {
		t.Fatalf("expected MoE metadata warning: %+v", in)
	}
}
