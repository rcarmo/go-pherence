package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestResolveSpecialTokenIDsFromTokenizer(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 4, UseImageStartEnd: true}
	tok := &config.MiniCPMVTokenizerMetadata{TokenIDs: map[string]int{"<im_start>": 10, "<im_end>": 11, "<im_patch>": 20, "<image>": 21}}
	ids, err := ResolveSpecialTokenIDs(summary, tok)
	if err != nil {
		t.Fatalf("ResolveSpecialTokenIDs: %v", err)
	}
	if ids.ImageStart != 10 || ids.ImageEnd != 11 || ids.ImagePatch != 20 || ids.Image != 21 || !ids.UseStartEnd {
		t.Fatalf("bad ids: %+v", ids)
	}
}

func TestBuildPromptPlanFromSummary(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 4, UseImageStartEnd: true}
	tok := &config.MiniCPMVTokenizerMetadata{TokenIDs: map[string]int{"<im_start>": 10, "<im_end>": 11, "<im_patch>": 20}}
	plan, ids, err := BuildPromptPlanFromSummary([]int{1, 10, 20, 20, 20, 20, 11, 2}, summary, tok)
	if err != nil {
		t.Fatalf("BuildPromptPlanFromSummary: %v", err)
	}
	if ids.ImagePatch != 20 || len(plan.ImageSpans) != 1 || plan.ImageSpans[0].PatchStart != 2 {
		t.Fatalf("bad plan ids=%+v plan=%+v", ids, plan)
	}
}

func TestResolveSpecialTokenIDsRejectsMissingPatch(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 4, UseImageStartEnd: false}
	if _, err := ResolveSpecialTokenIDs(summary, nil); err == nil {
		t.Fatalf("expected missing patch id to fail")
	}
}
