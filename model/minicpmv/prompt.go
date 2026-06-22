package minicpmv

import "fmt"

const (
	DefaultImagePatchToken = "<im_patch>"
	DefaultImageStartToken = "<im_start>"
	DefaultImageEndToken   = "<im_end>"
)

// PromptPlan describes where MiniCPM-V vision embeddings replace placeholder
// token embeddings. Upstream OpenBMB code inserts one resampler output sequence
// between <im_start> and <im_end>; the number of placeholder positions must
// equal num_query for each image.
type PromptPlan struct {
	InputIDs    []int
	ImageSpans  []ImageSpan
	NumQuery    int
	UseStartEnd bool
}

type ImageSpan struct {
	StartTokenPos int
	PatchStart    int
	PatchEnd      int
	EndTokenPos   int
}

// BuildPromptPlan validates MiniCPM-V multimodal token placement. It does not
// tokenize text; callers pass already-tokenized input IDs and the model-specific
// token IDs from the tokenizer/config. The returned spans are the positions that
// must be replaced by vision/resampler embeddings before language-model decode.
func BuildPromptPlan(inputIDs []int, numQuery, imagePatchToken, imageStartToken, imageEndToken int, useStartEnd bool) (PromptPlan, error) {
	plan := PromptPlan{InputIDs: append([]int(nil), inputIDs...), NumQuery: numQuery, UseStartEnd: useStartEnd}
	if numQuery <= 0 {
		return plan, fmt.Errorf("MiniCPM-V num_query must be positive, got %d", numQuery)
	}
	if !useStartEnd {
		return buildPatchOnlyPromptPlan(plan, imagePatchToken)
	}
	for i := 0; i < len(inputIDs); i++ {
		if inputIDs[i] != imageStartToken {
			continue
		}
		end := i + numQuery + 1
		if end >= len(inputIDs) {
			return plan, fmt.Errorf("MiniCPM-V image starting at token %d needs %d patch slots plus end token, input len=%d", i, numQuery, len(inputIDs))
		}
		for j := i + 1; j < end; j++ {
			if inputIDs[j] != imagePatchToken {
				return plan, fmt.Errorf("MiniCPM-V expected image patch token at %d, got %d", j, inputIDs[j])
			}
		}
		if inputIDs[end] != imageEndToken {
			return plan, fmt.Errorf("MiniCPM-V expected image end token at %d, got %d", end, inputIDs[end])
		}
		plan.ImageSpans = append(plan.ImageSpans, ImageSpan{StartTokenPos: i, PatchStart: i + 1, PatchEnd: end, EndTokenPos: end})
		i = end
	}
	if countToken(inputIDs, imageEndToken) != len(plan.ImageSpans) {
		return plan, fmt.Errorf("MiniCPM-V mismatched image start/end token counts")
	}
	return plan, nil
}

func buildPatchOnlyPromptPlan(plan PromptPlan, imagePatchToken int) (PromptPlan, error) {
	start := -1
	for i, tok := range plan.InputIDs {
		if tok == imagePatchToken {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			if err := appendPatchOnlySpan(&plan, start, i); err != nil {
				return plan, err
			}
			start = -1
		}
	}
	if start >= 0 {
		if err := appendPatchOnlySpan(&plan, start, len(plan.InputIDs)); err != nil {
			return plan, err
		}
	}
	return plan, nil
}

func appendPatchOnlySpan(plan *PromptPlan, start, end int) error {
	if end-start != plan.NumQuery {
		return fmt.Errorf("MiniCPM-V patch-only image span [%d,%d) has %d tokens, want num_query=%d", start, end, end-start, plan.NumQuery)
	}
	plan.ImageSpans = append(plan.ImageSpans, ImageSpan{StartTokenPos: -1, PatchStart: start, PatchEnd: end, EndTokenPos: -1})
	return nil
}

func countToken(ids []int, tok int) int {
	n := 0
	for _, id := range ids {
		if id == tok {
			n++
		}
	}
	return n
}

// ResamplerShape normalizes the OpenBMB perceiver-resampler dimensions. The
// upstream Resampler uses grid_size^2 learned queries, embed_dim equal to the
// language hidden size, num_heads=embed_dim/128 by default, and optional kv_proj
// when the vision tower dimension differs.
type ResamplerShape struct {
	GridSize          int
	NumQuery          int
	EmbedDim          int
	NumHeads          int
	KVDim             int
	NeedsKVProjection bool
}

func NewResamplerShape(numQuery, embedDim, numHeads, kvDim int) (ResamplerShape, error) {
	if numQuery <= 0 || embedDim <= 0 {
		return ResamplerShape{}, fmt.Errorf("MiniCPM-V invalid resampler dims num_query=%d embed_dim=%d", numQuery, embedDim)
	}
	grid := 0
	for i := 1; i*i <= numQuery; i++ {
		if i*i == numQuery {
			grid = i
			break
		}
	}
	if grid == 0 {
		return ResamplerShape{}, fmt.Errorf("MiniCPM-V num_query=%d is not square", numQuery)
	}
	if numHeads <= 0 {
		numHeads = embedDim / 128
	}
	if numHeads <= 0 || embedDim%numHeads != 0 {
		return ResamplerShape{}, fmt.Errorf("MiniCPM-V invalid resampler heads=%d for embed_dim=%d", numHeads, embedDim)
	}
	return ResamplerShape{GridSize: grid, NumQuery: numQuery, EmbedDim: embedDim, NumHeads: numHeads, KVDim: kvDim, NeedsKVProjection: kvDim > 0 && kvDim != embedDim}, nil
}
