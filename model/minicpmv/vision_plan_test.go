package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildVisionExecutionPlan(t *testing.T) {
	summary := config.MiniCPMVSummary{VisionModelType: "siglip_vision_model", ImageSize: 448, PatchSize: 14, VisionHiddenSize: 1152, VisionLayers: 27, NumQuery: 64, ResamplerGrid: 8, ResamplerHeads: 28, HiddenSize: 3584}
	inv := SummarizeTensors([]string{
		"vpm.embeddings.patch_embedding.weight",
		"vpm.encoder.layers.0.self_attn.q_proj.weight",
		"resampler.query.weight",
		"resampler.kv_proj.weight",
	})
	plan := BuildVisionExecutionPlan(summary, &inv)
	if plan.PatchGrid != 32 || plan.VisionTokens != 1024 || plan.ResamplerQuery != 64 || !plan.NeedsKVProj || plan.Ready {
		t.Fatalf("bad vision plan: %+v", plan)
	}
	if got := findVisionOp(plan, "patch_embedding"); got == nil || !got.Ready {
		t.Fatalf("patch op not ready: %+v", plan.Ops)
	}
	if got := findVisionOp(plan, "language_embedding_injection"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("injection op should be pending: %+v", plan.Ops)
	}
}

func TestBuildVisionExecutionPlanMissingTensors(t *testing.T) {
	summary := config.MiniCPMVSummary{VisionModelType: "siglip_vision_model", ImageSize: 448, PatchSize: 14, VisionHiddenSize: 1152, VisionLayers: 27, NumQuery: 64, ResamplerGrid: 8, ResamplerHeads: 28, HiddenSize: 3584}
	plan := BuildVisionExecutionPlan(summary, nil)
	if got := findVisionOp(plan, "patch_embedding"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("patch op should report missing tensors: %+v", plan.Ops)
	}
	if got := findVisionOp(plan, "resampler_queries"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("resampler op should report missing tensors: %+v", plan.Ops)
	}
}

func TestVisionModelTypeHelpers(t *testing.T) {
	if !IsLikelySigLIPVision(config.MiniCPMVSummary{VisionModelType: "SiglipVisionModel"}) {
		t.Fatalf("expected siglip detection")
	}
	if !IsLikelyEVAVision(config.MiniCPMVSummary{VisionModelType: "eva02"}) {
		t.Fatalf("expected eva detection")
	}
}

func findVisionOp(plan VisionExecutionPlan, name string) *VisionOp {
	for i := range plan.Ops {
		if plan.Ops[i].Name == name {
			return &plan.Ops[i]
		}
	}
	return nil
}
