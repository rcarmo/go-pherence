package minicpmv

import "fmt"

type AudioEmbeddingInjection struct {
	SequenceLength int `json:"sequence_length"`
	HiddenSize     int `json:"hidden_size"`
	Audios         int `json:"audios"`
	ReplacedTokens int `json:"replaced_tokens"`
}

func InjectAudioEmbeddings(tokenEmbeddings []float32, seqLen, hidden int, plan AudioPromptPlan, audioEmbeddings []float32) ([]float32, AudioEmbeddingInjection, error) {
	meta := AudioEmbeddingInjection{SequenceLength: seqLen, HiddenSize: hidden, Audios: len(plan.AudioSpans)}
	if seqLen <= 0 || hidden <= 0 {
		return nil, meta, fmt.Errorf("MiniCPM-O audio embedding injection: invalid seqLen=%d hidden=%d", seqLen, hidden)
	}
	if len(tokenEmbeddings) != seqLen*hidden {
		return nil, meta, fmt.Errorf("MiniCPM-O audio embedding injection: token embeddings len=%d want %d", len(tokenEmbeddings), seqLen*hidden)
	}
	if plan.PatchTokens <= 0 {
		return nil, meta, fmt.Errorf("MiniCPM-O audio embedding injection: invalid patch_tokens=%d", plan.PatchTokens)
	}
	wantAudio := len(plan.AudioSpans) * plan.PatchTokens * hidden
	if len(audioEmbeddings) != wantAudio {
		return nil, meta, fmt.Errorf("MiniCPM-O audio embedding injection: audio embeddings len=%d want %d", len(audioEmbeddings), wantAudio)
	}
	out := append([]float32(nil), tokenEmbeddings...)
	for audio, span := range plan.AudioSpans {
		if span.PatchStart < 0 || span.PatchEnd > seqLen || span.PatchEnd-span.PatchStart != plan.PatchTokens {
			return nil, meta, fmt.Errorf("MiniCPM-O audio embedding injection: invalid audio span %+v for seqLen=%d patch_tokens=%d", span, seqLen, plan.PatchTokens)
		}
		for q := 0; q < plan.PatchTokens; q++ {
			dst := (span.PatchStart + q) * hidden
			src := (audio*plan.PatchTokens + q) * hidden
			copy(out[dst:dst+hidden], audioEmbeddings[src:src+hidden])
			meta.ReplacedTokens++
		}
	}
	return out, meta, nil
}
