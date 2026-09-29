package nemotronasr

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// PCM32RNNT composes the released short-window encoder with one-stream greedy
// RNNT decoding. Its 5,160-sample input is a fixed prefix, not a streaming
// chunk or a whole-recording transcription; tokens retain blank IDs.
type PCM32RNNT struct {
	Projection *OfflineProjection
	Decoder    *RNNTDecoder
}

func LoadPCM32RNNT(file *safetensors.File) (*PCM32RNNT, error) {
	projection, err := LoadOfflineProjection(file)
	if err != nil {
		return nil, err
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		return nil, err
	}
	return &PCM32RNNT{Projection: projection, Decoder: decoder}, nil
}

// Decode returns owned raw token IDs and zero-based projected frame indices.
// Only lookahead 0 or 3 and a released prompt ID are accepted by the encoder.
func (m *PCM32RNNT) Decode(pcm []float32, lookahead, prompt int) (tokens, frames []int, err error) {
	return m.decode(pcm, lookahead, prompt, nil)
}

func (m *PCM32RNNT) decode(pcm []float32, lookahead, prompt int, observe func(int, []float32) error) (tokens, frames []int, err error) {
	if m == nil || m.Projection == nil || m.Decoder == nil || m.Projection.rnnt == nil {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR PCM decoder")
	}
	_, encoder, err := m.Projection.ForwardPCM32(pcm, lookahead, prompt)
	if err != nil {
		return nil, nil, err
	}
	return (&GreedyRNNT{Decoder: m.Decoder, Projection: m.Projection.rnnt}).decode(encoder, 5, observe)
}
