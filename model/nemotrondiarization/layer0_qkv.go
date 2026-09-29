package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Layer0QKV owns the input and first-layer normalisations plus pre-attention
// weights. Its outputs are pre-RoPE Q, K and V; attention and decoding are separate.
type Layer0QKV struct {
	inputGamma, inputBeta []float32
	gamma, beta           []float32
	q, k, v               []float32
}

func LoadLayer0QKV(file *safetensors.File) (*Layer0QKV, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	prefix := "model.audio_tower."
	load := func(name string, shape ...int) ([]float32, error) {
		values, got, err := file.GetFloat32(prefix + name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if len(got) != len(shape) {
			return nil, fmt.Errorf("%s: invalid rank %v", name, got)
		}
		size := 1
		for i, dim := range shape {
			if got[i] != dim {
				return nil, fmt.Errorf("%s: invalid shape %v", name, got)
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
	m := &Layer0QKV{}
	var err error
	if m.inputGamma, err = load("input_layer_norm.weight", projectedWidth); err != nil {
		return nil, err
	}
	if m.inputBeta, err = load("input_layer_norm.bias", projectedWidth); err != nil {
		return nil, err
	}
	if m.gamma, err = load("layers.0.layer_norm1.weight", projectedWidth); err != nil {
		return nil, err
	}
	if m.beta, err = load("layers.0.layer_norm1.bias", projectedWidth); err != nil {
		return nil, err
	}
	if m.q, err = load("layers.0.self_attn.q_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	if m.k, err = load("layers.0.self_attn.k_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	if m.v, err = load("layers.0.self_attn.v_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// Project accepts owned-or-caller [rows,512] stacking embeddings and returns
// independent, owned, row-major pre-RoPE [rows,512] Q/K/V buffers.
func (m *Layer0QKV) Project(input []float32, rows int) (q, k, v []float32, err error) {
	_, q, k, v, err = m.projectWithInputNormal(input, rows)
	return q, k, v, err
}

// projectWithInputNormal also returns the owned input-normalised residual
// needed by attention, avoiding a second normalisation of the same input.
func (m *Layer0QKV) projectWithInputNormal(input []float32, rows int) (inputNormal, q, k, v []float32, err error) {
	if m == nil || len(m.inputGamma) != projectedWidth || len(m.inputBeta) != projectedWidth || len(m.gamma) != projectedWidth || len(m.beta) != projectedWidth || len(m.q) != projectedWidth*projectedWidth || len(m.k) != projectedWidth*projectedWidth || len(m.v) != projectedWidth*projectedWidth {
		return nil, nil, nil, nil, fmt.Errorf("invalid Nemotron diarization layer-0 weights")
	}
	if rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, nil, nil, nil, fmt.Errorf("invalid Nemotron diarization layer-0 input shape")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, nil, nil, fmt.Errorf("non-finite Nemotron diarization layer-0 input")
		}
	}
	inputNormal = make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(inputNormal, input, rows, projectedWidth, m.inputGamma, m.inputBeta, 1e-5) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization input normalisation rejected shape")
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, inputNormal, rows, projectedWidth, m.gamma, m.beta, 1e-5) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization layer-0 normalisation rejected shape")
	}
	q, k, v = make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for _, item := range []struct {
		out, weight []float32
	}{{q, m.q}, {k, m.k}, {v, m.v}} {
		if !simd.DenseNTTo(item.out, normal, item.weight, rows, projectedWidth, projectedWidth, 1, projectedWidth, projectedWidth, projectedWidth) {
			return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization layer-0 projection rejected shape")
		}
	}
	return inputNormal, q, k, v, nil
}
