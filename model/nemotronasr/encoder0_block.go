package nemotronasr

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Encoder0Block owns the first released offline ASR encoder block's weights.
// It admits a bounded direct unmasked or zero/three-lookahead chunked window
// of 1..5 subsampled rows. Later blocks, caches and RNN-T are separate.
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
	return m.forwardOffline(input, rows, -1)
}

// ForwardOfflineLookahead applies the released chunk mask for zero or three
// lookahead tokens. No key/value or convolution padding cache is used.
func (m *Encoder0Block) ForwardOfflineLookahead(input []float32, rows, lookahead int) ([]float32, error) {
	if lookahead != 0 && lookahead != 3 {
		return nil, fmt.Errorf("unsupported Nemotron ASR encoder-0 lookahead")
	}
	return m.forwardOffline(input, rows, lookahead)
}

func (m *Encoder0Block) forwardOffline(input []float32, rows, lookahead int) ([]float32, error) {
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
	var attentionResidual []float32
	if lookahead < 0 {
		_, attentionResidual, err = m.Attention.ForwardOffline(ff1, rows)
	} else {
		_, attentionResidual, err = m.Attention.ForwardOfflineLookahead(ff1, rows, lookahead)
	}
	if err != nil {
		return nil, err
	}
	_, convResidual, err := m.Conv.ForwardOffline(attentionResidual, rows)
	if err != nil {
		return nil, err
	}
	return m.FF2.ForwardOffline(convResidual, rows)
}
