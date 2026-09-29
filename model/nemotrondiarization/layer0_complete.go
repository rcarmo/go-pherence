package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const diarizationIntermediate = 2048

// Layer0Complete owns the released first audio-transformer layer. The offline
// operation is unmasked bidirectional attention followed by a GELU MLP. Later
// layers, streaming caches, upsampling and speaker classification are separate.
type Layer0Complete struct {
	attention            *Layer0Attention
	normWeight, normBias []float32
	fc1Weight, fc1Bias   []float32
	fc2Weight, fc2Bias   []float32
	fc1Packed, fc2Packed []float32
}

func LoadLayer0Complete(file *safetensors.File) (*Layer0Complete, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	attention, err := LoadLayer0Attention(file)
	if err != nil {
		return nil, err
	}
	prefix := "model.audio_tower.layers.0."
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
	m := &Layer0Complete{attention: attention}
	if m.normWeight, err = load("layer_norm2.weight", projectedWidth); err != nil {
		return nil, err
	}
	if m.normBias, err = load("layer_norm2.bias", projectedWidth); err != nil {
		return nil, err
	}
	if m.fc1Weight, err = load("mlp.fc1.weight", diarizationIntermediate, projectedWidth); err != nil {
		return nil, err
	}
	if m.fc1Bias, err = load("mlp.fc1.bias", diarizationIntermediate); err != nil {
		return nil, err
	}
	if m.fc2Weight, err = load("mlp.fc2.weight", projectedWidth, diarizationIntermediate); err != nil {
		return nil, err
	}
	if m.fc2Bias, err = load("mlp.fc2.bias", projectedWidth); err != nil {
		return nil, err
	}
	if simd.HasSgemmAsm {
		if m.fc1Packed, err = simd.PackSgemmNTWeights(m.fc1Weight, diarizationIntermediate, projectedWidth, projectedWidth); err != nil {
			return nil, err
		}
		if m.fc2Packed, err = simd.PackSgemmNTWeights(m.fc2Weight, projectedWidth, diarizationIntermediate, diarizationIntermediate); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// ForwardOffline returns an owned layer-0 output from an entire unmasked
// [rows,512] stacking window, with positions beginning at zero.
func (m *Layer0Complete) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || m.attention == nil || len(m.normWeight) != projectedWidth || len(m.normBias) != projectedWidth || len(m.fc1Weight) != diarizationIntermediate*projectedWidth || len(m.fc1Bias) != diarizationIntermediate || len(m.fc2Weight) != projectedWidth*diarizationIntermediate || len(m.fc2Bias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization layer-0 model")
	}
	_, residual, err := m.attention.ForwardOffline(input, rows)
	if err != nil {
		return nil, err
	}
	normal := make([]float32, len(residual))
	if !simd.LayerNormLastAxisTo(normal, residual, rows, projectedWidth, m.normWeight, m.normBias, 1e-5) {
		return nil, fmt.Errorf("Nemotron diarization layer-0 MLP normalisation rejected shape")
	}
	intermediate := make([]float32, rows*diarizationIntermediate)
	if !diarizationDenseMLP(intermediate, normal, m.fc1Weight, m.fc1Packed, rows, diarizationIntermediate, projectedWidth) {
		return nil, fmt.Errorf("Nemotron diarization layer-0 fc1 rejected shape")
	}
	for i, value := range intermediate {
		intermediate[i] = value + m.fc1Bias[i%diarizationIntermediate]
	}
	if !simd.GELUErfF32To(intermediate, intermediate) {
		return nil, fmt.Errorf("Nemotron diarization layer-0 GELU rejected shape")
	}
	output := make([]float32, len(residual))
	if !diarizationDenseMLP(output, intermediate, m.fc2Weight, m.fc2Packed, rows, projectedWidth, diarizationIntermediate) {
		return nil, fmt.Errorf("Nemotron diarization layer-0 fc2 rejected shape")
	}
	for i, value := range output {
		output[i] = residual[i] + value + m.fc2Bias[i%projectedWidth]
	}
	return output, nil
}
