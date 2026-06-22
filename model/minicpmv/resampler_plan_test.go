package minicpmv

import "testing"

func TestClassifyResamplerTensorName(t *testing.T) {
	cases := map[string]ResamplerTensorRole{
		"resampler.query.weight":       ResamplerQuery,
		"resampler.pos_embed":          ResamplerPosEmbed,
		"resampler.kv_proj.weight":     ResamplerKVProj,
		"resampler.attn.q_proj.weight": ResamplerQProj,
		"resampler.attn.k_proj.weight": ResamplerKProj,
		"resampler.attn.v_proj.weight": ResamplerVProj,
		"resampler.attn.o_proj.weight": ResamplerOProj,
		"resampler.ln_q.weight":        ResamplerNorm,
		"resampler.mlp.fc1.weight":     ResamplerMLP,
		"mm_projector.0.weight":        ResamplerKVProj,
	}
	for name, want := range cases {
		if got := ClassifyResamplerTensorName(name); got != want {
			t.Fatalf("ClassifyResamplerTensorName(%q)=%s want %s", name, got, want)
		}
	}
}

func TestBuildResamplerTensorPlanReadyWithKVProjection(t *testing.T) {
	plan := BuildResamplerTensorPlan([]string{
		"resampler.query.weight",
		"resampler.attn.q_proj.weight",
		"resampler.attn.k_proj.weight",
		"resampler.attn.v_proj.weight",
		"resampler.attn.o_proj.weight",
		"resampler.kv_proj.weight",
		"llm.model.embed_tokens.weight",
	}, true)
	if !plan.Ready || len(plan.MissingRequired) != 0 || plan.Counts[ResamplerQuery] != 1 || plan.Counts[ResamplerKVProj] != 1 {
		t.Fatalf("bad ready resampler plan: %+v", plan)
	}
	if len(plan.Bindings) != 6 {
		t.Fatalf("bindings=%d want 6: %+v", len(plan.Bindings), plan.Bindings)
	}
}

func TestBuildResamplerTensorPlanMissingKVProjection(t *testing.T) {
	plan := BuildResamplerTensorPlan([]string{"resampler.query.weight"}, true)
	if plan.Ready || len(plan.MissingRequired) != 1 || plan.MissingRequired[0] != ResamplerKVProj {
		t.Fatalf("expected missing kv projection: %+v", plan)
	}
}

func TestBuildResamplerTensorPlanNoKVProjectionRequired(t *testing.T) {
	plan := BuildResamplerTensorPlan([]string{"resampler.query.weight"}, false)
	if !plan.Ready || len(plan.MissingRequired) != 0 {
		t.Fatalf("expected query-only ready when kv projection not needed: %+v", plan)
	}
}
