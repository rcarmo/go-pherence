package minicpmv

import "fmt"

type MultiModalEmbeddingPlan struct {
	SequenceLength      int `json:"sequence_length"`
	HiddenSize          int `json:"hidden_size"`
	Images              int `json:"images"`
	Audios              int `json:"audios"`
	ImageReplacedTokens int `json:"image_replaced_tokens"`
	AudioReplacedTokens int `json:"audio_replaced_tokens"`
	TotalReplacedTokens int `json:"total_replaced_tokens"`
}

func BuildMultiModalEmbeddingPlan(seqLen, hidden int, imagePlan *PromptPlan, audioPlan *AudioPromptPlan) (MultiModalEmbeddingPlan, error) {
	out := MultiModalEmbeddingPlan{SequenceLength: seqLen, HiddenSize: hidden}
	if seqLen <= 0 || hidden <= 0 {
		return out, fmt.Errorf("MiniCPM-V/O multimodal embedding plan: invalid seqLen=%d hidden=%d", seqLen, hidden)
	}
	if imagePlan != nil {
		out.Images = len(imagePlan.ImageSpans)
		for _, span := range imagePlan.ImageSpans {
			if span.PatchStart < 0 || span.PatchEnd > seqLen || span.PatchEnd-span.PatchStart != imagePlan.NumQuery {
				return out, fmt.Errorf("MiniCPM-V/O multimodal embedding plan: invalid image span %+v", span)
			}
			out.ImageReplacedTokens += span.PatchEnd - span.PatchStart
		}
	}
	if audioPlan != nil {
		out.Audios = len(audioPlan.AudioSpans)
		for _, span := range audioPlan.AudioSpans {
			if span.PatchStart < 0 || span.PatchEnd > seqLen || span.PatchEnd-span.PatchStart != audioPlan.PatchTokens {
				return out, fmt.Errorf("MiniCPM-V/O multimodal embedding plan: invalid audio span %+v", span)
			}
			out.AudioReplacedTokens += span.PatchEnd - span.PatchStart
		}
	}
	out.TotalReplacedTokens = out.ImageReplacedTokens + out.AudioReplacedTokens
	return out, nil
}
