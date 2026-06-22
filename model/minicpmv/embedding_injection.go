package minicpmv

import "fmt"

type EmbeddingInjection struct {
	SequenceLength int `json:"sequence_length"`
	HiddenSize     int `json:"hidden_size"`
	Images         int `json:"images"`
	ReplacedTokens int `json:"replaced_tokens"`
}

// InjectImageEmbeddings returns a copy of tokenEmbeddings where every planned
// image patch span has been replaced by the corresponding resampler outputs.
// Layout is row-major [sequence][hidden] for tokenEmbeddings and
// [image][num_query][hidden] flattened row-major for imageEmbeddings.
func InjectImageEmbeddings(tokenEmbeddings []float32, seqLen, hidden int, plan PromptPlan, imageEmbeddings []float32) ([]float32, EmbeddingInjection, error) {
	meta := EmbeddingInjection{SequenceLength: seqLen, HiddenSize: hidden, Images: len(plan.ImageSpans)}
	if seqLen <= 0 || hidden <= 0 {
		return nil, meta, fmt.Errorf("MiniCPM-V/O embedding injection: invalid seqLen=%d hidden=%d", seqLen, hidden)
	}
	if len(tokenEmbeddings) != seqLen*hidden {
		return nil, meta, fmt.Errorf("MiniCPM-V/O embedding injection: token embeddings len=%d want %d", len(tokenEmbeddings), seqLen*hidden)
	}
	if plan.NumQuery <= 0 {
		return nil, meta, fmt.Errorf("MiniCPM-V/O embedding injection: invalid num_query=%d", plan.NumQuery)
	}
	wantImage := len(plan.ImageSpans) * plan.NumQuery * hidden
	if len(imageEmbeddings) != wantImage {
		return nil, meta, fmt.Errorf("MiniCPM-V/O embedding injection: image embeddings len=%d want %d", len(imageEmbeddings), wantImage)
	}
	out := append([]float32(nil), tokenEmbeddings...)
	for img, span := range plan.ImageSpans {
		if span.PatchStart < 0 || span.PatchEnd > seqLen || span.PatchEnd-span.PatchStart != plan.NumQuery {
			return nil, meta, fmt.Errorf("MiniCPM-V/O embedding injection: invalid image span %+v for seqLen=%d num_query=%d", span, seqLen, plan.NumQuery)
		}
		for q := 0; q < plan.NumQuery; q++ {
			dst := (span.PatchStart + q) * hidden
			src := (img*plan.NumQuery + q) * hidden
			copy(out[dst:dst+hidden], imageEmbeddings[src:src+hidden])
			meta.ReplacedTokens++
		}
	}
	return out, meta, nil
}
