package nemotrondiarization

import (
	"fmt"
	"math"
)

// SpeakerCacheUncompressed holds one stream's arrival-order speaker frames
// and recent FIFO frames. It stops before the released score-based compressor.
type SpeakerCacheUncompressed struct {
	speaker, speakerProbs, fifo []float32
}

// Snapshot returns owned [speaker,512], [speaker,8], and [fifo,512] arrays.
func (s *SpeakerCacheUncompressed) Snapshot() (speaker, probs, fifo []float32) {
	if s == nil {
		return nil, nil, nil
	}
	return append([]float32(nil), s.speaker...), append([]float32(nil), s.speakerProbs...), append([]float32(nil), s.fifo...)
}

// Prepare returns owned cache+FIFO+chunk embeddings; the chunk includes
// lookahead, which Update excludes from retained state.
func (s *SpeakerCacheUncompressed) Prepare(chunk []float32, frames, lookahead int) ([]float32, error) {
	if err := s.validate(chunk, frames, lookahead); err != nil {
		return nil, err
	}
	input := make([]float32, 0, len(s.speaker)+len(s.fifo)+len(chunk))
	input = append(input, s.speaker...)
	input = append(input, s.fifo...)
	return append(input, chunk...), nil
}

func (s *SpeakerCacheUncompressed) validate(chunk []float32, frames, lookahead int) error {
	if s == nil || frames < 1 || frames > diarizationStreamFIFO || lookahead < 0 || lookahead > 4 || len(chunk) != (frames+lookahead)*projectedWidth || len(s.speaker)%projectedWidth != 0 || len(s.fifo)%projectedWidth != 0 || len(s.speaker)/projectedWidth > diarizationStreamFIFO || len(s.fifo)/projectedWidth > diarizationStreamFIFO || len(s.speakerProbs) != len(s.speaker)/projectedWidth*diarizationSpeakers {
		return fmt.Errorf("invalid Nemotron diarization uncompressed cache")
	}
	for _, values := range [][]float32{chunk, s.speaker, s.fifo, s.speakerProbs} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("non-finite Nemotron diarization cache input")
			}
		}
	}
	return nil
}

// Update consumes logits and a validity mask for the prepared full context.
// On first FIFO overflow it moves at least 222 oldest frames to the
// speaker cache. When that cache would exceed 264 frames, it returns an error
// without changing any state; score-based compression is not qualified here.
func (s *SpeakerCacheUncompressed) Update(chunk []float32, frames, lookahead int, logits []float32, mask []bool) error {
	if err := s.validate(chunk, frames, lookahead); err != nil {
		return err
	}
	cachedFrames := (len(s.speaker) + len(s.fifo)) / projectedWidth
	contextFrames := cachedFrames + frames + lookahead
	if len(mask) != contextFrames {
		return fmt.Errorf("invalid Nemotron diarization cache mask")
	}
	pooled, err := PoolSpeakerProbabilities(logits, mask)
	if err != nil {
		return err
	}
	fifo := make([]float32, 0, len(s.fifo)+frames*projectedWidth)
	fifo = append(fifo, s.fifo...)
	fifo = append(fifo, chunk[:frames*projectedWidth]...)
	pop := 0
	if len(fifo)/projectedWidth > diarizationStreamFIFO {
		pop = max(222, len(fifo)/projectedWidth-diarizationStreamFIFO)
		pop = min(pop, len(fifo)/projectedWidth)
	}
	if len(s.speaker)/projectedWidth+pop > diarizationStreamFIFO {
		return fmt.Errorf("Nemotron diarization speaker compression not qualified")
	}
	if pop > 0 {
		speaker := append(append([]float32(nil), s.speaker...), fifo[:pop*projectedWidth]...)
		// Before compression, the reference re-estimates the probabilities of
		// existing speaker frames on this update rather than using stored ones.
		probs := append([]float32(nil), pooled[:(len(s.speaker)/projectedWidth+pop)*diarizationSpeakers]...)
		s.speaker, s.speakerProbs = speaker, probs
	}
	s.fifo = append([]float32(nil), fifo[pop*projectedWidth:]...)
	return nil
}
