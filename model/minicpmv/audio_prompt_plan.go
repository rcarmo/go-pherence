package minicpmv

import "fmt"

type AudioPromptPlan struct {
	InputIDs    []int       `json:"input_ids,omitempty"`
	AudioSpans  []AudioSpan `json:"audio_spans"`
	PatchTokens int         `json:"patch_tokens"`
	UseStartEnd bool        `json:"use_start_end"`
}

type AudioSpan struct {
	StartTokenPos int `json:"start_token_pos"`
	PatchStart    int `json:"patch_start"`
	PatchEnd      int `json:"patch_end"`
	EndTokenPos   int `json:"end_token_pos"`
}

func BuildAudioPromptPlan(inputIDs []int, patchTokens, audioPatchToken, audioStartToken, audioEndToken int, useStartEnd bool) (AudioPromptPlan, error) {
	plan := AudioPromptPlan{InputIDs: append([]int(nil), inputIDs...), PatchTokens: patchTokens, UseStartEnd: useStartEnd}
	if patchTokens <= 0 {
		return plan, fmt.Errorf("MiniCPM-O audio patch token count must be positive, got %d", patchTokens)
	}
	if !useStartEnd {
		return buildAudioPatchOnlyPromptPlan(plan, audioPatchToken)
	}
	for i := 0; i < len(inputIDs); i++ {
		if inputIDs[i] != audioStartToken {
			continue
		}
		end := i + patchTokens + 1
		if end >= len(inputIDs) {
			return plan, fmt.Errorf("MiniCPM-O audio starting at token %d needs %d patch slots plus end token, input len=%d", i, patchTokens, len(inputIDs))
		}
		for j := i + 1; j < end; j++ {
			if inputIDs[j] != audioPatchToken {
				return plan, fmt.Errorf("MiniCPM-O expected audio patch token at %d, got %d", j, inputIDs[j])
			}
		}
		if inputIDs[end] != audioEndToken {
			return plan, fmt.Errorf("MiniCPM-O expected audio end token at %d, got %d", end, inputIDs[end])
		}
		plan.AudioSpans = append(plan.AudioSpans, AudioSpan{StartTokenPos: i, PatchStart: i + 1, PatchEnd: end, EndTokenPos: end})
		i = end
	}
	if countToken(inputIDs, audioEndToken) != len(plan.AudioSpans) {
		return plan, fmt.Errorf("MiniCPM-O mismatched audio start/end token counts")
	}
	return plan, nil
}

func buildAudioPatchOnlyPromptPlan(plan AudioPromptPlan, audioPatchToken int) (AudioPromptPlan, error) {
	start := -1
	for i, tok := range plan.InputIDs {
		if tok == audioPatchToken {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			if err := appendAudioPatchOnlySpan(&plan, start, i); err != nil {
				return plan, err
			}
			start = -1
		}
	}
	if start >= 0 {
		if err := appendAudioPatchOnlySpan(&plan, start, len(plan.InputIDs)); err != nil {
			return plan, err
		}
	}
	return plan, nil
}

func appendAudioPatchOnlySpan(plan *AudioPromptPlan, start, end int) error {
	if end-start != plan.PatchTokens {
		return fmt.Errorf("MiniCPM-O patch-only audio span [%d,%d) has %d tokens, want patch_tokens=%d", start, end, end-start, plan.PatchTokens)
	}
	plan.AudioSpans = append(plan.AudioSpans, AudioSpan{StartTokenPos: -1, PatchStart: start, PatchEnd: end, EndTokenPos: -1})
	return nil
}
