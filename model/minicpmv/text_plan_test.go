package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildTextExecutionPlan(t *testing.T) {
	summary := config.MiniCPMVSummary{TextModelType: "qwen2", HiddenSize: 4, Layers: 2, Heads: 1, KVHeads: 1, HeadDim: 4, VocabSize: 100, IntermediateSize: 8}
	gen := &config.MiniCPMVGenerationConfig{MaxNewTokens: 64}
	inv := SummarizeTensors([]string{
		"llm.model.embed_tokens.weight",
		"llm.model.layers.0.self_attn.q_proj.weight",
		"llm.lm_head.weight",
	})
	plan := BuildTextExecutionPlan(summary, gen, &inv)
	if !plan.MetadataReady || !plan.TensorReady || !plan.Generation || !plan.HasEmbedding || !plan.HasLayers || !plan.HasLMHead || plan.Ready {
		t.Fatalf("bad text plan: %+v", plan)
	}
	if got := findTextOp(plan, "prefill_decode"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("prefill should be pending: %+v", plan.Ops)
	}
}

func TestBuildTextExecutionPlanMissingTensors(t *testing.T) {
	summary := config.MiniCPMVSummary{HiddenSize: 4, Layers: 2, Heads: 1, VocabSize: 100}
	plan := BuildTextExecutionPlan(summary, nil, nil)
	if !plan.MetadataReady || plan.TensorReady || plan.Generation || plan.Ready {
		t.Fatalf("unexpected ready text plan: %+v", plan)
	}
	if got := findTextOp(plan, "token_embedding"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("embedding op should be missing: %+v", plan.Ops)
	}
}

func findTextOp(plan TextExecutionPlan, name string) *TextOp {
	for i := range plan.Ops {
		if plan.Ops[i].Name == name {
			return &plan.Ops[i]
		}
	}
	return nil
}
