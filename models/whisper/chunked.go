package whisper

import (
	"github.com/rcarmo/go-pherence/loader/audio"
)

// ChunkedTranscribe handles audio longer than 30s by splitting into overlapping chunks.
// Each chunk is 30s with configurable overlap for boundary continuity.
func (w *Whisper) ChunkedTranscribe(samples []float32, overlapSec float64) ([]Segment, error) {
	cfg := w.Config
	chunkSamples := 30 * 16000 // 30 seconds at 16kHz
	if overlapSec <= 0 {
		overlapSec = 1.0
	}
	overlapSamples := int(overlapSec * 16000)

	if len(samples) <= chunkSamples {
		// Single chunk
		segs := w.transcribeChunk(samples, 0)
		return segs, nil
	}

	var allSegments []Segment
	offset := 0
	chunkIdx := 0

	for offset < len(samples) {
		end := offset + chunkSamples
		if end > len(samples) {
			end = len(samples)
		}

		chunk := samples[offset:end]
		timeOffset := float64(offset) / 16000.0

		segs := w.transcribeChunk(chunk, timeOffset)
		allSegments = appendNonOverlapping(allSegments, segs)

		// Advance by chunk minus overlap
		offset += chunkSamples - overlapSamples
		chunkIdx++

		// Safety: don't process more than 100 chunks (~50 minutes)
		if chunkIdx > 100 {
			break
		}
	}

	return mergeAdjacentSegments(allSegments, cfg), nil
}

// transcribeChunk transcribes a single ≤30s audio chunk with timestamp decoding.
func (w *Whisper) transcribeChunk(samples []float32, timeOffset float64) []Segment {
	cfg := w.Config
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    cfg.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil || len(mel[0]) == 0 {
		return nil
	}

	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel

	state := NewDecoderState(cfg, encoderOutput, encLen, w.Decoder)
	segments := GreedyDecodeWithTimestamps(w.Decoder, state, cfg)

	// Adjust timestamps by chunk offset
	for i := range segments {
		segments[i].Start += timeOffset
		segments[i].End += timeOffset
	}

	return segments
}

// appendNonOverlapping appends new segments, skipping any that overlap with existing ones.
func appendNonOverlapping(existing, new []Segment) []Segment {
	if len(existing) == 0 {
		return new
	}
	lastEnd := existing[len(existing)-1].End

	for _, seg := range new {
		if seg.Start >= lastEnd-0.1 { // 100ms tolerance
			existing = append(existing, seg)
		}
	}
	return existing
}

// mergeAdjacentSegments merges segments with the same text or very close timestamps.
func mergeAdjacentSegments(segments []Segment, cfg Config) []Segment {
	if len(segments) <= 1 {
		return segments
	}
	_ = cfg

	var merged []Segment
	current := segments[0]

	for _, seg := range segments[1:] {
		// Merge if gap < 50ms and same text content
		if seg.Start-current.End < 0.05 && seg.Text == current.Text {
			current.End = seg.End
			current.Tokens = append(current.Tokens, seg.Tokens...)
		} else {
			merged = append(merged, current)
			current = seg
		}
	}
	merged = append(merged, current)
	return merged
}
