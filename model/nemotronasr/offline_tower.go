package nemotronasr

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// OfflineEncoderTower owns the released 24-layer ASR encoder weights. It
// accepts only a complete five-row subsampled window at lookahead 0 or 3;
// it does not implement streaming caches or RNN-T decoding.
type OfflineEncoderTower struct {
	layers [24]*Encoder0Block
}

func LoadOfflineEncoderTower(file *safetensors.File) (*OfflineEncoderTower, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	m := &OfflineEncoderTower{}
	var err error
	if m.layers[0], err = LoadEncoder0Block(file); err != nil {
		return nil, err
	}
	for layer := 1; layer < len(m.layers); layer++ {
		if m.layers[layer], err = LoadIndexedEncoderBlock(file, layer); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// ForwardOfflineLookahead returns owned [5,1024] encoder output. The caller's
// projected subsampling input remains unchanged.
func (m *OfflineEncoderTower) ForwardOfflineLookahead(input []float32, lookahead int) ([]float32, error) {
	if m == nil || len(input) != 5*encoderWidth || (lookahead != 0 && lookahead != 3) {
		return nil, fmt.Errorf("invalid Nemotron ASR offline tower input")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR tower input")
		}
	}
	hidden := input
	for layer, block := range m.layers {
		if block == nil {
			return nil, fmt.Errorf("missing Nemotron ASR encoder layer %d", layer)
		}
		var err error
		hidden, err = block.ForwardOfflineLookahead(hidden, 5, lookahead)
		if err != nil {
			return nil, fmt.Errorf("Nemotron ASR encoder layer %d: %w", layer, err)
		}
	}
	for _, value := range hidden {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR tower output")
		}
	}
	return hidden, nil
}
