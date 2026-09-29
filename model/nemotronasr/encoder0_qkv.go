package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Encoder0QKV owns the first ASR encoder attention's pre-normalisation and
// bias-free Q/K/V projections. Relative positional attention, cache, output
// projection, convolution and RNN-T are separate unimplemented contracts.
type Encoder0QKV struct {
	gamma, beta []float32
	q, k, v     []float32
}

func LoadEncoder0QKV(file *safetensors.File) (*Encoder0QKV, error) {
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
	m := &Encoder0QKV{}
	var err error
	if m.gamma, err = load("norm_self_att.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.beta, err = load("norm_self_att.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.q, err = load("self_attn.q_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if m.k, err = load("self_attn.k_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if m.v, err = load("self_attn.v_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// Project accepts a complete [rows,1024] first-FF1 residual and returns
// owned, independent, pre-attention Q/K/V arrays in row-major order.
func (m *Encoder0QKV) Project(input []float32, rows int) (q, k, v []float32, err error) {
	if m == nil || len(m.gamma) != encoderWidth || len(m.beta) != encoderWidth || len(m.q) != encoderWidth*encoderWidth || len(m.k) != encoderWidth*encoderWidth || len(m.v) != encoderWidth*encoderWidth {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 QKV weights")
	}
	if rows < 1 || rows > 17 || len(input) != rows*encoderWidth {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 QKV input shape")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 QKV input")
		}
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, rows, encoderWidth, m.gamma, m.beta, 1e-5) {
		return nil, nil, nil, fmt.Errorf("Nemotron ASR encoder-0 attention norm rejected shape")
	}
	q, k, v = make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for _, item := range []struct{ out, weight []float32 }{{q, m.q}, {k, m.k}, {v, m.v}} {
		if !simd.DenseNTTo(item.out, normal, item.weight, rows, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
			return nil, nil, nil, fmt.Errorf("Nemotron ASR encoder-0 QKV projection rejected shape")
		}
	}
	return q, k, v, nil
}
