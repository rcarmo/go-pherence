package nemotronasr

import (
	"fmt"
	"math"
)

// GreedyRNNT composes a single-stream predictor and joint network over at
// most 139 already prompt-projected encoder rows. It returns owned token and
// frame-index slices; blank tokens are retained for reference comparison.
// No tokenizer or native long-window/streaming encoder is implemented here.
type GreedyRNNT struct {
	Decoder    *RNNTDecoder
	Projection *RNNTProjection
}

const maxQualifiedRNNTFrames = 139

// Decode uses the released blank=13087 and max_symbols_per_step=10 rules.
func (m *GreedyRNNT) Decode(encoder []float32, rows int) (tokens, frames []int, err error) {
	return m.decode(encoder, rows, nil)
}

// decode accepts an optional per-step observer for independent logit checks.
func (m *GreedyRNNT) decode(encoder []float32, rows int, observe func(int, []float32) error) (tokens, frames []int, err error) {
	if m == nil || m.Decoder == nil || m.Projection == nil || rows < 1 || rows > maxQualifiedRNNTFrames || len(encoder) != rows*rnntHidden {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR greedy input")
	}
	for _, value := range encoder {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR greedy input")
		}
	}
	var state DecoderState
	logitsScratch := make([]float32, rnntVocabulary)
	activatedScratch := make([]float32, rnntHidden)
	frame, token, symbols := 0, rnntBlank, 0
	for frame < rows {
		// A frame can emit at most ten symbols; at the bound the reference
		// advances the frame even when the last symbol is non-blank.
		if len(tokens) >= rows*10 {
			return nil, nil, fmt.Errorf("Nemotron ASR greedy bound exceeded")
		}
		decoded, err := m.Decoder.stepBorrowed(token, &state)
		if err != nil {
			return nil, nil, err
		}
		if err := m.Projection.jointTo(logitsScratch, activatedScratch, encoder[frame*rnntHidden:(frame+1)*rnntHidden], decoded, 1); err != nil {
			return nil, nil, err
		}
		if observe != nil {
			// Preserve the observer's owned per-step logits while Decode
			// reuses its scratch for the next decision.
			if err := observe(len(tokens), append([]float32(nil), logitsScratch...)); err != nil {
				return nil, nil, err
			}
		}
		selected := 0
		for i := 1; i < len(logitsScratch); i++ {
			if logitsScratch[i] > logitsScratch[selected] {
				selected = i
			}
		}
		tokens = append(tokens, selected)
		frames = append(frames, frame)
		token = selected
		if advanceGreedyFrame(token, &symbols) {
			frame++
		}
	}
	return tokens, frames, nil
}

func advanceGreedyFrame(token int, symbols *int) bool {
	if token == rnntBlank {
		*symbols = 0
		return true
	}
	*symbols++
	if *symbols >= 10 {
		*symbols = 0
		return true
	}
	return false
}
