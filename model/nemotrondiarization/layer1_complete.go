package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Layer1Complete owns the released second audio-transformer layer. The offline
// operation is unmasked bidirectional attention followed by a GELU MLP. Remaining
// layers, streaming caches, upsampling and speaker classification are separate.
type Layer1Complete struct {
	attention            *Layer1Attention
	normWeight, normBias []float32
	fc1Weight, fc1Bias   []float32
	fc2Weight, fc2Bias   []float32
}

func LoadLayer1Complete(file *safetensors.File) (*Layer1Complete, error) {
	return LoadIndexedAudioLayer(file, 1)
}

// LoadIndexedAudioLayer loads one offline audio-transformer layer after layer 0.
// Loading does not qualify its outputs; test a full-context reference window.
func LoadIndexedAudioLayer(file *safetensors.File, layer int) (*Layer1Complete, error) {
	if file == nil || layer < 1 || layer >= 31 {
		return nil, fmt.Errorf("invalid Nemotron diarization layer or checkpoint")
	}
	attention, err := loadIndexedAttention(file, layer)
	if err != nil {
		return nil, err
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
	m := &Layer1Complete{attention: attention}
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
	return m, nil
}

// ForwardOffline returns an owned layer-1 output from an entire unmasked
// [rows,512] layer-0 output, with positions beginning at zero.
func (m *Layer1Complete) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || m.attention == nil || len(m.normWeight) != projectedWidth || len(m.normBias) != projectedWidth || len(m.fc1Weight) != diarizationIntermediate*projectedWidth || len(m.fc1Bias) != diarizationIntermediate || len(m.fc2Weight) != projectedWidth*diarizationIntermediate || len(m.fc2Bias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization layer-1 model")
	}
	_, residual, err := m.attention.ForwardOffline(input, rows)
	if err != nil {
		return nil, err
	}
	normal := make([]float32, len(residual))
	if !simd.LayerNormLastAxisTo(normal, residual, rows, projectedWidth, m.normWeight, m.normBias, 1e-5) {
		return nil, fmt.Errorf("Nemotron diarization layer-1 MLP normalisation rejected shape")
	}
	intermediate := make([]float32, rows*diarizationIntermediate)
	if !simd.DenseNTTo(intermediate, normal, m.fc1Weight, rows, diarizationIntermediate, projectedWidth, 1, projectedWidth, projectedWidth, diarizationIntermediate) {
		return nil, fmt.Errorf("Nemotron diarization layer-1 fc1 rejected shape")
	}
	for i, value := range intermediate {
		intermediate[i] = value + m.fc1Bias[i%diarizationIntermediate]
	}
	if !simd.GELUErfF32To(intermediate, intermediate) {
		return nil, fmt.Errorf("Nemotron diarization layer-1 GELU rejected shape")
	}
	output := make([]float32, len(residual))
	if !simd.DenseNTTo(output, intermediate, m.fc2Weight, rows, projectedWidth, diarizationIntermediate, 1, diarizationIntermediate, diarizationIntermediate, projectedWidth) {
		return nil, fmt.Errorf("Nemotron diarization layer-1 fc2 rejected shape")
	}
	for i, value := range output {
		output[i] = residual[i] + value + m.fc2Bias[i%projectedWidth]
		if math.IsNaN(float64(output[i])) || math.IsInf(float64(output[i]), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization layer-1 output")
		}
	}
	return output, nil
}
