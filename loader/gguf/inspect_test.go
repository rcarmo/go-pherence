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

func TestInspectOpenWarnsMissingMoEMetadata(t *testing.T) {
	g := &GGUF{Meta: map[string]any{"general.architecture": "qwen3moe"}, Tensors: []TensorInfo{{Name: "blk.0.ffn_gate_exps.weight", QType: QuantQ4_K}}}
	in := InspectOpen("model.gguf", g)
	if !in.HasMoE || len(in.ReadinessWarnings) == 0 {
		t.Fatalf("expected MoE metadata warning: %+v", in)
	}
}
