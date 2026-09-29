package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Encoder0FeedForward2 owns the second feed-forward and output-normalisation
// weights for the first released ASR encoder block. Input is the convolution
// residual; later blocks, caches and RNN-T are separate contracts.
type Encoder0FeedForward2 struct {
	gamma, beta       []float32
	first, last       []float32
	outGamma, outBeta []float32
}

func LoadEncoder0FeedForward2(file *safetensors.File) (*Encoder0FeedForward2, error) {
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
	m := &Encoder0FeedForward2{}
	var err error
	if m.gamma, err = load("norm_feed_forward2.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.beta, err = load("norm_feed_forward2.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.first, err = load("feed_forward2.linear1.weight", encoderFFWidth, encoderWidth); err != nil {
		return nil, err
	}
	if m.last, err = load("feed_forward2.linear2.weight", encoderWidth, encoderFFWidth); err != nil {
		return nil, err
	}
	if m.outGamma, err = load("norm_out.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.outBeta, err = load("norm_out.bias", encoderWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns an owned [rows,1024] post-block output. Both FF2
// linear projections are bias-free in this checkpoint; eval dropout is off.
func (m *Encoder0FeedForward2) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || len(m.gamma) != encoderWidth || len(m.beta) != encoderWidth || len(m.first) != encoderFFWidth*encoderWidth || len(m.last) != encoderWidth*encoderFFWidth || len(m.outGamma) != encoderWidth || len(m.outBeta) != encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 FF2 weights")
	}
	if rows < 1 || rows > 5 || len(input) != rows*encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder-0 FF2 input shape")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 FF2 input")
		}
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, rows, encoderWidth, m.gamma, m.beta, 1e-5) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF2 normalisation rejected shape")
	}
	intermediate := make([]float32, rows*encoderFFWidth)
	if !simd.DenseNTTo(intermediate, normal, m.first, rows, encoderFFWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderFFWidth) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF2 linear1 rejected shape")
	}
	if !simd.SiLUTo(intermediate, intermediate) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF2 activation rejected shape")
	}
	output := make([]float32, len(input))
	if !simd.DenseNTTo(output, intermediate, m.last, rows, encoderWidth, encoderFFWidth, 1, encoderFFWidth, encoderFFWidth, encoderWidth) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 FF2 linear2 rejected shape")
	}
	for i, value := range output {
		output[i] = input[i] + 0.5*value
	}
	final := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(final, output, rows, encoderWidth, m.outGamma, m.outBeta, 1e-5) {
		return nil, fmt.Errorf("Nemotron ASR encoder-0 output normalisation rejected shape")
	}
	return final, nil
}
