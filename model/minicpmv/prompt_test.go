package minicpmv

import "testing"

func TestBuildPromptPlanStartEnd(t *testing.T) {
	ids := []int{1, 10, 20, 20, 20, 20, 11, 2, 10, 20, 20, 20, 20, 11}
	plan, err := BuildPromptPlan(ids, 4, 20, 10, 11, true)
	if err != nil {
		t.Fatalf("BuildPromptPlan: %v", err)
	}
	if len(plan.ImageSpans) != 2 {
		t.Fatalf("spans=%d want 2", len(plan.ImageSpans))
	}
	if plan.ImageSpans[0].PatchStart != 2 || plan.ImageSpans[0].PatchEnd != 6 || plan.ImageSpans[0].EndTokenPos != 6 {
		t.Fatalf("bad first span: %+v", plan.ImageSpans[0])
	}
	if plan.ImageSpans[1].StartTokenPos != 8 || plan.ImageSpans[1].PatchEnd != 13 {
		t.Fatalf("bad second span: %+v", plan.ImageSpans[1])
	}
}

func TestBuildPromptPlanRejectsBadEndToken(t *testing.T) {
	ids := []int{10, 20, 20, 20, 20, 12}
	if _, err := BuildPromptPlan(ids, 4, 20, 10, 11, true); err == nil {
		t.Fatalf("expected bad end token to fail")
	}
}

func TestBuildPromptPlanPatchOnly(t *testing.T) {
	ids := []int{1, 20, 20, 20, 20, 2}
	plan, err := BuildPromptPlan(ids, 4, 20, 10, 11, false)
	if err != nil {
		t.Fatalf("BuildPromptPlan patch-only: %v", err)
	}
	if len(plan.ImageSpans) != 1 || plan.ImageSpans[0].PatchStart != 1 || plan.ImageSpans[0].PatchEnd != 5 {
		t.Fatalf("bad patch-only span: %+v", plan.ImageSpans)
	}
}

func TestNewResamplerShape(t *testing.T) {
	shape, err := NewResamplerShape(64, 4096, 0, 1792)
	if err != nil {
		t.Fatalf("NewResamplerShape: %v", err)
	}
	if shape.GridSize != 8 || shape.NumHeads != 32 || !shape.NeedsKVProjection {
		t.Fatalf("unexpected shape: %+v", shape)
	}
}

func TestNewResamplerShapeRejectsNonSquareQuery(t *testing.T) {
	if _, err := NewResamplerShape(63, 4096, 0, 4096); err == nil {
		t.Fatalf("expected non-square query count to fail")
	}
}
