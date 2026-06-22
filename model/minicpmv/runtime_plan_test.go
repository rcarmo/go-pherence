package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildRuntimePlanMetadataReadyButRuntimePending(t *testing.T) {
	summary := config.MiniCPMVSummary{HiddenSize: 3584, Layers: 28, Heads: 28, NumQuery: 64, ResamplerGrid: 8, ImageSize: 448, PatchSize: 14, UseImageStartEnd: true}
	processor := &config.MiniCPMVProcessorConfig{NormalizedSize: 448, PatchSize: 14}
	tokenizer := &config.MiniCPMVTokenizerMetadata{TokenIDs: map[string]int{"<im_start>": 10, "<im_end>": 11, "<im_patch>": 20}}
	inv := SummarizeTensors([]string{
		"llm.model.embed_tokens.weight",
		"llm.model.layers.0.self_attn.q_proj.weight",
		"vpm.encoder.layers.0.self_attn.q_proj.weight",
		"resampler.query.weight",
	})
	plan := BuildRuntimePlan(summary, processor, tokenizer, &inv)
	if !plan.ConfigReady || !plan.ProcessorReady || !plan.TokenizerReady || !plan.SpecialTokensReady || !plan.TensorMetadataReady || !plan.ImagePreprocessReady || !plan.PromptPlanningReady {
		t.Fatalf("expected metadata scaffolds ready: %+v", plan)
	}
	if plan.RuntimeReady {
		t.Fatalf("runtime must remain pending until tensor execution is implemented")
	}
	if got := findOp(plan, "vision_tower_execution"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("missing pending vision op: %+v", plan.Ops)
	}
}

func TestBuildRuntimePlanReportsMissingTokenizer(t *testing.T) {
	summary := config.MiniCPMVSummary{HiddenSize: 3584, Layers: 28, Heads: 28, NumQuery: 64, ResamplerGrid: 8, ImageSize: 448, PatchSize: 14, UseImageStartEnd: true, ImageTokenID: 20}
	plan := BuildRuntimePlan(summary, nil, nil, nil)
	if plan.TokenizerReady || plan.SpecialTokensReady || plan.TensorMetadataReady || plan.RuntimeReady {
		t.Fatalf("unexpected ready flags: %+v", plan)
	}
	if got := findOp(plan, "special_tokens"); got == nil || got.Reason == "" {
		t.Fatalf("missing special token reason: %+v", plan.Ops)
	}
}

func findOp(plan RuntimePlan, name string) *RuntimeOpStatus {
	for i := range plan.Ops {
		if plan.Ops[i].Name == name {
			return &plan.Ops[i]
		}
	}
	return nil
}
