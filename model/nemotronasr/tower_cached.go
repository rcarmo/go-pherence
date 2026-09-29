package nemotronasr

import (
	"fmt"
	"math"
)

// CachedEncoderTower owns the 24 per-layer attention and causal convolution
// states for one prepared-input stream. The weights are immutable and may be
// shared; this state is mutable and cannot be shared across concurrent calls.
// It does not perform PCM scheduling, subsampling, or RNNT decoding.
type CachedEncoderTower struct {
	Tower  *OfflineEncoderTower
	states [24]Encoder0ChunkState
	seen   int
}

// ForwardChunk consumes one to five fully valid projected encoder rows. The
// input must already include the released encoder input scale. All 24 cache
// transitions commit together only after a finite output succeeds.
func (s *CachedEncoderTower) ForwardChunk(input []float32, rows, lookahead int) ([]float32, error) {
	if s == nil || s.Tower == nil || rows < 1 || rows > 5 || len(input) != rows*encoderWidth || (lookahead != 0 && lookahead != 3) || s.seen < 0 || s.seen > int(^uint(0)>>1)-rows {
		return nil, fmt.Errorf("invalid Nemotron ASR cached tower input")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR cached tower input")
		}
	}
	states := s.states
	hidden := input
	for layer, block := range s.Tower.layers {
		if block == nil {
			return nil, fmt.Errorf("missing Nemotron ASR cached encoder layer %d", layer)
		}
		var err error
		hidden, err = block.ForwardCachedChunk(hidden, rows, lookahead, &states[layer])
		if err != nil {
			return nil, fmt.Errorf("Nemotron ASR cached encoder layer %d: %w", layer, err)
		}
	}
	s.states = states
	s.seen += rows
	return hidden, nil
}
