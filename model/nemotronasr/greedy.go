package nemotronasr

import (
	"fmt"
	"math"
)

// GreedyRNNT composes a single-stream predictor and joint network over at
// most five already prompt-projected encoder rows. It returns owned token and
// frame-index slices; blank tokens are retained for reference comparison.
// No tokenizer, longer recording, or streaming encoder is implemented here.
type GreedyRNNT struct {
	Decoder    *RNNTDecoder
	Projection *RNNTProjection
}

// Decode uses the released blank=13087 and max_symbols_per_step=10 rules.
func (m *GreedyRNNT) Decode(encoder []float32, rows int) (tokens, frames []int, err error) {
	return m.decode(encoder, rows, nil)
}

// decode accepts an optional per-step observer for independent logit checks.
func (m *GreedyRNNT) decode(encoder []float32, rows int, observe func(int, []float32) error) (tokens, frames []int, err error) {
	if m == nil || m.Decoder == nil || m.Projection == nil || rows < 1 || rows > 5 || len(encoder) != rows*rnntHidden {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR greedy input")
	}
	for _, value := range encoder {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR greedy input")
		}
	}
	var state DecoderState
	frame, token, symbols := 0, rnntBlank, 0
	for frame < rows {
		// A frame can emit at most ten symbols; at the bound the reference
		// advances the frame even when the last symbol is non-blank.
		if len(tokens) >= rows*10 {
			return nil, nil, fmt.Errorf("Nemotron ASR greedy bound exceeded")
		}
		decoded, err := m.Decoder.Step(token, &state)
		if err != nil {
			return nil, nil, err
		}
		logits, err := m.Projection.Joint(encoder[frame*rnntHidden:(frame+1)*rnntHidden], decoded, 1)
		if err != nil {
			return nil, nil, err
		}
		if observe != nil {
			if err := observe(len(tokens), logits); err != nil {
				return nil, nil, err
			}
		}
		selected := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[selected] {
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
