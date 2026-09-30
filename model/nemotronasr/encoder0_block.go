package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
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

// Encoder0ChunkState owns layer-0 attention and convolution history for one
// stream. It is mutable and must not be shared concurrently.
type Encoder0ChunkState struct {
	Attention Encoder0KVCache
	Conv      Encoder0ConvCache
}

// ForwardCachedChunk composes FF1, cached attention, causal convolution and
// FF2 over one to five current rows. The two caches commit together only
// after a finite output has been produced. This is not an integrated encoder.
func (m *Encoder0Block) ForwardCachedChunk(input []float32, rows, lookahead int, state *Encoder0ChunkState) ([]float32, error) {
	if m == nil || m.FF1 == nil || m.Attention == nil || m.Conv == nil || m.FF2 == nil || state == nil {
		return nil, fmt.Errorf("invalid Nemotron ASR cached encoder-0 block")
	}
	prepared := *state
	ff1, err := m.FF1.ForwardOffline(input, rows)
	if err != nil {
		return nil, err
	}
	if m.Attention.qkv == nil || len(m.Attention.qkv.gamma) != encoderWidth || len(m.Attention.qkv.beta) != encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR attention normalisation")
	}
	normal := make([]float32, len(ff1))
	if !simd.LayerNormLastAxisTo(normal, ff1, rows, encoderWidth, m.Attention.qkv.gamma, m.Attention.qkv.beta, 1e-5) {
		return nil, fmt.Errorf("Nemotron ASR attention normalisation rejected shape")
	}
	attention, err := m.Attention.ForwardCachedChunk(normal, rows, lookahead, &prepared.Attention)
	if err != nil {
		return nil, err
	}
	// Attention returned an owned output. This block only consumes its
	// residual, so accumulate FF1 there instead of allocating another row.
	attentionResidual := attention
	for i, value := range attentionResidual {
		attentionResidual[i] = ff1[i] + value
		if math.IsNaN(float64(attentionResidual[i])) || math.IsInf(float64(attentionResidual[i]), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR attention residual")
		}
	}
	_, convResidual, err := m.Conv.ForwardCachedChunk(attentionResidual, rows, &prepared.Conv)
	if err != nil {
		return nil, err
	}
	output, err := m.FF2.ForwardOffline(convResidual, rows)
	if err != nil {
		return nil, err
	}
	for _, value := range output {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR cached block output")
		}
	}
	*state = prepared
	return output, nil
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
