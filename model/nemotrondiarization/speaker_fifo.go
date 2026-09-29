package nemotrondiarization

import (
	"fmt"
	"math"
)

const diarizationStreamFIFO = 264

// SpeakerFIFO owns a single stream's recent encoder embeddings before the
// released cache's speaker-compression threshold is reached. Compression and
// speaker-cache probabilities require separate validation.
type SpeakerFIFO struct {
	fifo []float32 // [frames,512], oldest first
}

func (s *SpeakerFIFO) Snapshot() []float32 {
	if s == nil {
		return nil
	}
	return append([]float32(nil), s.fifo...)
}

// Prepare returns owned cached+current+lookahead embeddings without changing
// the FIFO. Update then commits only the current frames, excluding lookahead.
func (s *SpeakerFIFO) Prepare(chunk []float32, frames, lookahead int) ([]float32, error) {
	if s == nil || frames < 1 || lookahead < 0 || lookahead > 4 || len(chunk) != (frames+lookahead)*projectedWidth || len(s.fifo)%projectedWidth != 0 || len(s.fifo)/projectedWidth > diarizationStreamFIFO {
		return nil, fmt.Errorf("invalid Nemotron diarization FIFO input")
	}
	for _, value := range chunk {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization FIFO input")
		}
	}
	return append(append([]float32(nil), s.fifo...), chunk...), nil
}

func (s *SpeakerFIFO) Update(chunk []float32, frames, lookahead int) error {
	if _, err := s.Prepare(chunk, frames, lookahead); err != nil {
		return err
	}
	if len(s.fifo)/projectedWidth+frames > diarizationStreamFIFO {
		return fmt.Errorf("Nemotron diarization FIFO compression not qualified")
	}
	s.fifo = append(append([]float32(nil), s.fifo...), chunk[:frames*projectedWidth]...)
	return nil
}
