package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const (
	encoderWidth   = 1024
	encoderFFWidth = 4096
)

// Encoder0FeedForward1 owns the released first encoder block's first
// feed-forward weights. Attention, causal convolution, subsequent layers and
// RNN-T decoding are outside this bounded offline operator.
type Encoder0FeedForward1 struct {
	gamma, beta []float32
	first, last []float32
}

func LoadEncoder0FeedForward1(file *safetensors.File) (*Encoder0FeedForward1, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	prefix := "encoder.layers.0."
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(prefix + name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("%s: invalid rank %v", name, shape)
		}
		size := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("%s: invalid shape %v", name, shape)
			}
			size *= dim
		}
		if len(values) != size {
			return nil, fmt.Errorf("%s: invalid element count", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("%s: non-finite weight", name)
			}
		}
		return values, nil
	}
	m := &Encoder0FeedForward1{}
	var err error
	if m.gamma, err = load("norm_feed_forward1.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.beta, err = load("norm_feed_forward1.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.first, err = load("feed_forward1.linear1.weight", encoderFFWidth, encoderWidth); err != nil {
		return nil, err
	}
	if m.last, err = load("feed_forward1.linear2.weight", encoderWidth, encoderFFWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns owned [rows,1024] embeddings for the first FF1
// residual. This exact released encoder uses bias-free projections and SiLU.
// Eval dropout is disabled. Caller input remains untouched.
func (m *Encoder0FeedForward1) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || len(m.gamma) != encoderWidth || len(m.beta) != encoderWidth || len(m.first) != encoderFFWidth*encoderWidth || len(m.last) != encoderWidth*encoderFFWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 FF1 weights")
	}
	if rows < 1 || rows > 17 || len(input) != rows*encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 FF1 input shape")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 FF1 input")
		}
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, rows, encoderWidth, m.gamma, m.beta, 1e-5) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF1 normalisation rejected shape")
	}
	intermediate := make([]float32, rows*encoderFFWidth)
	if !simd.DenseNTTo(intermediate, normal, m.first, rows, encoderFFWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderFFWidth) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF1 linear1 rejected shape")
	}
	if !simd.SiLUTo(intermediate, intermediate) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF1 activation rejected shape")
	}
	output := make([]float32, len(input))
	if !simd.DenseNTTo(output, intermediate, m.last, rows, encoderWidth, encoderFFWidth, 1, encoderFFWidth, encoderFFWidth, encoderWidth) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF1 linear2 rejected shape")
	}
	for i, value := range output {
		output[i] = input[i] + 0.5*value
	}
	return output, nil
}
