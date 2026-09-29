package nemotrondiarization

import (
	"fmt"
	"math"
)

// SpeakerCache owns one stream's compressed or uncompressed arrival-order
// speaker embeddings/probabilities and recent FIFO frames. It consumes
// prepared encoder embeddings and logits; model-level streaming is separate.
type SpeakerCache struct {
	compressor                  *SpeakerCompressor
	speaker, speakerProbs, fifo []float32
	compressed                  bool
}

func NewSpeakerCache(compressor *SpeakerCompressor) (*SpeakerCache, error) {
	if compressor == nil || len(compressor.silence) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization compressor")
	}
	return &SpeakerCache{compressor: compressor}, nil
}

func (s *SpeakerCache) Snapshot() (speaker, probs, fifo []float32, compressed bool) {
	if s == nil {
		return nil, nil, nil, false
	}
	return append([]float32(nil), s.speaker...), append([]float32(nil), s.speakerProbs...), append([]float32(nil), s.fifo...), s.compressed
}

func (s *SpeakerCache) Prepare(chunk []float32, frames, lookahead int) ([]float32, error) {
	if err := s.validate(chunk, frames, lookahead); err != nil {
		return nil, err
	}
	out := make([]float32, 0, len(s.speaker)+len(s.fifo)+len(chunk))
	out = append(out, s.speaker...)
	out = append(out, s.fifo...)
	return append(out, chunk...), nil
}

func (s *SpeakerCache) validate(chunk []float32, frames, lookahead int) error {
	if s == nil || s.compressor == nil || len(s.compressor.silence) != projectedWidth || frames < 1 || frames > diarizationStreamFIFO || lookahead < 0 || lookahead > 4 || len(chunk) != (frames+lookahead)*projectedWidth || len(s.speaker)%projectedWidth != 0 || len(s.fifo)%projectedWidth != 0 || len(s.speaker)/projectedWidth > diarizationStreamFIFO || len(s.fifo)/projectedWidth > diarizationStreamFIFO || len(s.speakerProbs) != len(s.speaker)/projectedWidth*diarizationSpeakers {
		return fmt.Errorf("invalid Nemotron diarization speaker cache")
	}
	for _, values := range [][]float32{chunk, s.speaker, s.fifo, s.speakerProbs} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("non-finite Nemotron diarization speaker cache")
			}
		}
	}
	return nil
}

// Update commits only after preparing finite new state. The reference
// recomputes plain speaker probabilities until compression; afterward it
// retains stored speaker probabilities while re-estimating FIFO probabilities.
func (s *SpeakerCache) Update(chunk []float32, frames, lookahead int, logits []float32, mask []bool) error {
	if err := s.validate(chunk, frames, lookahead); err != nil {
		return err
	}
	cacheFrames := len(s.speaker) / projectedWidth
	fifoFrames := len(s.fifo) / projectedWidth
	contextFrames := cacheFrames + fifoFrames + frames + lookahead
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
		pop = min(max(222, len(fifo)/projectedWidth-diarizationStreamFIFO), len(fifo)/projectedWidth)
	}
	speaker := s.speaker
	probs := s.speakerProbs
	compressed := s.compressed
	if pop > 0 {
		speaker = append(append([]float32(nil), s.speaker...), fifo[:pop*projectedWidth]...)
		stored := pooled[:cacheFrames*diarizationSpeakers]
		if s.compressed {
			stored = s.speakerProbs
		}
		probs = make([]float32, 0, (cacheFrames+pop)*diarizationSpeakers)
		probs = append(probs, stored...)
		probs = append(probs, pooled[cacheFrames*diarizationSpeakers:(cacheFrames+pop)*diarizationSpeakers]...)
		if len(speaker)/projectedWidth > diarizationStreamFIFO {
			if len(speaker)/projectedWidth > 528 {
				return fmt.Errorf("Nemotron diarization compression window too large")
			}
			speaker, probs, err = s.compressor.Compress(speaker, probs)
			if err != nil {
				return err
			}
			compressed = true
		}
	}
	s.speaker = append([]float32(nil), speaker...)
	s.speakerProbs = append([]float32(nil), probs...)
	s.fifo = append([]float32(nil), fifo[pop*projectedWidth:]...)
	s.compressed = compressed
	return nil
}
