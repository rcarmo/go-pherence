package minicpmv

import "testing"

func TestBuildAudioPromptPlanStartEnd(t *testing.T) {
	ids := []int{1, 30, 40, 40, 40, 31, 2}
	plan, err := BuildAudioPromptPlan(ids, 3, 40, 30, 31, true)
	if err != nil {
		t.Fatalf("BuildAudioPromptPlan: %v", err)
	}
	if len(plan.AudioSpans) != 1 || plan.AudioSpans[0].PatchStart != 2 || plan.AudioSpans[0].PatchEnd != 5 || plan.AudioSpans[0].EndTokenPos != 5 {
		t.Fatalf("bad audio span: %+v", plan)
	}
}

func TestBuildAudioPromptPlanRejectsBadPatch(t *testing.T) {
	ids := []int{30, 40, 41, 40, 31}
	if _, err := BuildAudioPromptPlan(ids, 3, 40, 30, 31, true); err == nil {
		t.Fatalf("expected bad patch token to fail")
	}
}

func TestBuildAudioPromptPlanPatchOnly(t *testing.T) {
	ids := []int{1, 40, 40, 40, 2}
	plan, err := BuildAudioPromptPlan(ids, 3, 40, 30, 31, false)
	if err != nil {
		t.Fatalf("BuildAudioPromptPlan patch-only: %v", err)
	}
	if len(plan.AudioSpans) != 1 || plan.AudioSpans[0].PatchStart != 1 || plan.AudioSpans[0].PatchEnd != 4 {
		t.Fatalf("bad patch-only span: %+v", plan)
	}
}

func TestBuildAudioPromptPlanRejectsMismatchedEnd(t *testing.T) {
	ids := []int{30, 40, 40, 40, 32}
	if _, err := BuildAudioPromptPlan(ids, 3, 40, 30, 31, true); err == nil {
		t.Fatalf("expected mismatched end token to fail")
	}
}
