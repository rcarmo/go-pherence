package diffusiongemma

import "testing"

func TestBuildForwardBufferPlanExpertsScratchCoversNormedHiddenRows(t *testing.T) {
	shape := Shape{
		CanvasLength:        3,
		TextHiddenSize:      16,
		VocabSize:           32,
		NumExperts:          4,
		TopKExperts:         1,
		MoEIntermediateSize: 5,
	}
	plan := BuildForwardBufferPlan(shape)
	wantHiddenRows := shape.CanvasLength * shape.TextHiddenSize
	if plan.Experts < wantHiddenRows {
		t.Fatalf("Experts scratch=%d, want at least hidden rows=%d", plan.Experts, wantHiddenRows)
	}
	if plan.Experts != wantHiddenRows {
		t.Fatalf("Experts scratch=%d, want hidden rows=%d when hidden rows exceed topK intermediate", plan.Experts, wantHiddenRows)
	}
}

func TestBuildForwardBufferPlanExpertsScratchKeepsLargerExpertActivation(t *testing.T) {
	shape := Shape{
		CanvasLength:        3,
		TextHiddenSize:      8,
		VocabSize:           32,
		NumExperts:          4,
		TopKExperts:         4,
		MoEIntermediateSize: 7,
	}
	plan := BuildForwardBufferPlan(shape)
	wantExpert := shape.CanvasLength * shape.TopKExperts * shape.MoEIntermediateSize
	if plan.Experts != wantExpert {
		t.Fatalf("Experts scratch=%d, want expert activation size=%d", plan.Experts, wantExpert)
	}
}
