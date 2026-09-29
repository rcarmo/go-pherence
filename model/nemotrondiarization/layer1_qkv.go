package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Layer1QKV owns the second audio layer's pre-attention normalisation and
// bias-free Q/K/V projections. Its input must be the complete first layer's
// output; this slice does not implement layer-1 attention or streaming.
type Layer1QKV struct {
	gamma, beta []float32
	q, k, v     []float32
}

func LoadLayer1QKV(file *safetensors.File) (*Layer1QKV, error) {
	return loadIndexedQKV(file, 1)
}

func loadIndexedQKV(file *safetensors.File, layer int) (*Layer1QKV, error) {
	if file == nil || layer < 1 || layer >= 31 {
		return nil, fmt.Errorf("invalid Nemotron diarization layer or checkpoint")
	}
	prefix := fmt.Sprintf("model.audio_tower.layers.%d.", layer)
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
	m := &Layer1QKV{}
	var err error
	if m.gamma, err = load("layer_norm1.weight", projectedWidth); err != nil {
		return nil, err
	}
	if m.beta, err = load("layer_norm1.bias", projectedWidth); err != nil {
		return nil, err
	}
	if m.q, err = load("self_attn.q_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	if m.k, err = load("self_attn.k_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	if m.v, err = load("self_attn.v_proj.weight", projectedWidth, projectedWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// Project returns independently owned, pre-RoPE [rows,512] Q/K/V arrays.
func (m *Layer1QKV) Project(input []float32, rows int) (q, k, v []float32, err error) {
	if m == nil || len(m.gamma) != projectedWidth || len(m.beta) != projectedWidth || len(m.q) != projectedWidth*projectedWidth || len(m.k) != projectedWidth*projectedWidth || len(m.v) != projectedWidth*projectedWidth {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron diarization layer-1 weights")
	}
	if rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron diarization layer-1 input shape")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron diarization layer-1 input")
		}
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, rows, projectedWidth, m.gamma, m.beta, 1e-5) {
		return nil, nil, nil, fmt.Errorf("Nemotron diarization layer-1 normalisation rejected shape")
	}
	q, k, v = make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for _, item := range []struct{ out, weight []float32 }{{q, m.q}, {k, m.k}, {v, m.v}} {
		if !simd.DenseNTTo(item.out, normal, item.weight, rows, projectedWidth, projectedWidth, 1, projectedWidth, projectedWidth, projectedWidth) {
			return nil, nil, nil, fmt.Errorf("Nemotron diarization layer-1 projection rejected shape")
		}
	}
	return q, k, v, nil
}
