package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// OfflineAudioTower runs the released 31-layer audio transformer on a complete
// bidirectional stacking window. It does not implement the frontend,
// upsampler, speaker head, or a streaming cache.
type OfflineAudioTower struct {
	first                  *Layer0Complete
	remaining              []*Layer1Complete
	finalWeight, finalBias []float32
}

func LoadOfflineAudioTower(file *safetensors.File) (*OfflineAudioTower, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	first, err := LoadLayer0Complete(file)
	if err != nil {
		return nil, err
	}
	tower := &OfflineAudioTower{first: first, remaining: make([]*Layer1Complete, 30)}
	for index := range tower.remaining {
		tower.remaining[index], err = LoadIndexedAudioLayer(file, index+1)
		if err != nil {
			return nil, err
		}
	}
	load := func(name string) ([]float32, error) {
		values, shape, err := file.GetFloat32("model.audio_tower.layer_norm." + name)
		if err != nil {
			return nil, err
		}
		if len(shape) != 1 || shape[0] != projectedWidth || len(values) != projectedWidth {
			return nil, fmt.Errorf("invalid Nemotron diarization tower norm %s shape %v", name, shape)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron diarization tower norm %s", name)
			}
		}
		return values, nil
	}
	if tower.finalWeight, err = load("weight"); err != nil {
		return nil, err
	}
	if tower.finalBias, err = load("bias"); err != nil {
		return nil, err
	}
	return tower, nil
}

// ForwardOffline returns an owned, final-normalised [rows,512] window.
// It must receive the complete stacking context used by the reference.
func (m *OfflineAudioTower) ForwardOffline(input []float32, rows int) ([]float32, error) {
	if m == nil || m.first == nil || len(m.remaining) != 30 || len(m.finalWeight) != projectedWidth || len(m.finalBias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization audio tower")
	}
	hidden, err := m.first.ForwardOffline(input, rows)
	if err != nil {
		return nil, err
	}
	for _, layer := range m.remaining {
		if layer == nil {
			return nil, fmt.Errorf("missing Nemotron diarization audio layer")
		}
		hidden, err = layer.ForwardOffline(hidden, rows)
		if err != nil {
			return nil, err
		}
	}
	output := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(output, hidden, rows, projectedWidth, m.finalWeight, m.finalBias, 1e-5) {
		return nil, fmt.Errorf("Nemotron diarization final norm rejected shape")
	}
	for _, value := range output {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization tower output")
		}
	}
	return output, nil
}
