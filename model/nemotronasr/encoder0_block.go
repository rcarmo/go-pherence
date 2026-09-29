package nemotronasr

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Encoder0Block owns the first released offline ASR encoder block's weights.
// Only an unmasked window of 1..5 subsampled rows at default lookahead 3 is
// admitted; later blocks, caches and RNN-T are not implemented here.
type Encoder0Block struct {
	FF1       *Encoder0FeedForward1
	Attention *Encoder0Attention
	Conv      *Encoder0Convolution
	FF2       *Encoder0FeedForward2
}

func LoadEncoder0Block(file *safetensors.File) (*Encoder0Block, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	m := &Encoder0Block{}
	var err error
	if m.FF1, err = LoadEncoder0FeedForward1(file); err != nil {
		return nil, err
	}
	if m.Attention, err = LoadEncoder0Attention(file); err != nil {
		return nil, err
	}
	if m.Conv, err = LoadEncoder0Convolution(file); err != nil {
		return nil, err
	}
	if m.FF2, err = LoadEncoder0FeedForward2(file); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns independent owned block output, preserving caller
// input. The nested modules validate finite inputs and their own shapes.
func (m *Encoder0Block) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || m.FF1 == nil || m.Attention == nil || m.Conv == nil || m.FF2 == nil {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 block")
	}
	if rows < 1 || rows > 5 || len(input) != rows*encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 block window")
	}
	ff1, err := m.FF1.ForwardOffline(input, rows)
	if err != nil {
		return nil, err
	}
	_, attentionResidual, err := m.Attention.ForwardOffline(ff1, rows)
	if err != nil {
		return nil, err
	}
	_, convResidual, err := m.Conv.ForwardOffline(attentionResidual, rows)
	if err != nil {
		return nil, err
	}
	return m.FF2.ForwardOffline(convResidual, rows)
}
