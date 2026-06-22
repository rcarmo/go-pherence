package minicpmv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/config"
)

type SpecialTokenIDs struct {
	Image       int  `json:"image"`
	ImageStart  int  `json:"image_start"`
	ImageEnd    int  `json:"image_end"`
	ImagePatch  int  `json:"image_patch"`
	UseStartEnd bool `json:"use_start_end"`
}

func ResolveSpecialTokenIDs(summary config.MiniCPMVSummary, tok *config.MiniCPMVTokenizerMetadata) (SpecialTokenIDs, error) {
	out := SpecialTokenIDs{Image: summary.ImageTokenID, ImageStart: summary.ImageStartTokenID, ImageEnd: summary.ImageEndTokenID, ImagePatch: summary.ImageTokenID, UseStartEnd: summary.UseImageStartEnd}
	if tok != nil {
		if id, ok := lookupAny(tok.TokenIDs, "<im_patch>", "<image>"); ok {
			out.ImagePatch = id
		}
		if id, ok := lookupAny(tok.TokenIDs, "<image>"); ok && out.Image == 0 {
			out.Image = id
		}
		if id, ok := lookupAny(tok.TokenIDs, "<im_start>", "<|im_start|>"); ok {
			out.ImageStart = id
		}
		if id, ok := lookupAny(tok.TokenIDs, "<im_end>", "<|im_end|>"); ok {
			out.ImageEnd = id
		}
	}
	if out.ImagePatch == 0 {
		return out, fmt.Errorf("MiniCPM-V/O image patch token id is missing")
	}
	if out.UseStartEnd && (out.ImageStart == 0 || out.ImageEnd == 0) {
		return out, fmt.Errorf("MiniCPM-V/O image start/end token ids are missing")
	}
	return out, nil
}

func BuildPromptPlanFromSummary(inputIDs []int, summary config.MiniCPMVSummary, tok *config.MiniCPMVTokenizerMetadata) (PromptPlan, SpecialTokenIDs, error) {
	ids, err := ResolveSpecialTokenIDs(summary, tok)
	if err != nil {
		return PromptPlan{}, ids, err
	}
	plan, err := BuildPromptPlan(inputIDs, summary.NumQuery, ids.ImagePatch, ids.ImageStart, ids.ImageEnd, ids.UseStartEnd)
	return plan, ids, err
}

func lookupAny(m map[string]int, keys ...string) (int, bool) {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != 0 {
			return v, true
		}
	}
	return 0, false
}
