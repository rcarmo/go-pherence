package minicpmv

import "testing"

func TestBuildMultiModalEmbeddingPlan(t *testing.T) {
	img := &PromptPlan{NumQuery: 2, ImageSpans: []ImageSpan{{PatchStart: 1, PatchEnd: 3}}}
	aud := &AudioPromptPlan{PatchTokens: 3, AudioSpans: []AudioSpan{{PatchStart: 4, PatchEnd: 7}}}
	plan, err := BuildMultiModalEmbeddingPlan(8, 4, img, aud)
	if err != nil {
		t.Fatalf("BuildMultiModalEmbeddingPlan: %v", err)
	}
	if plan.Images != 1 || plan.Audios != 1 || plan.ImageReplacedTokens != 2 || plan.AudioReplacedTokens != 3 || plan.TotalReplacedTokens != 5 {
		t.Fatalf("bad multimodal embedding plan: %+v", plan)
	}
}

func TestBuildMultiModalEmbeddingPlanRejectsBadSpan(t *testing.T) {
	img := &PromptPlan{NumQuery: 2, ImageSpans: []ImageSpan{{PatchStart: 1, PatchEnd: 4}}}
	if _, err := BuildMultiModalEmbeddingPlan(8, 4, img, nil); err == nil {
		t.Fatalf("expected bad image span")
	}
	aud := &AudioPromptPlan{PatchTokens: 2, AudioSpans: []AudioSpan{{PatchStart: 7, PatchEnd: 10}}}
	if _, err := BuildMultiModalEmbeddingPlan(8, 4, nil, aud); err == nil {
		t.Fatalf("expected bad audio span")
	}
}
