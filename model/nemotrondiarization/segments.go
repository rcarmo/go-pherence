package nemotrondiarization

import (
	"fmt"
	"math"
)

// Segment uses the released processor's two-decimal second boundaries and
// zero-based speaker indices. Overlapping speakers yield overlapping spans.
type Segment struct {
	Start   float64 `json:"Start"`
	End     float64 `json:"End"`
	Speaker int     `json:"Speaker"`
}

// ExtractSegments applies the reference's sigmoid>0.5 activity rule to
// [frames,8] raw logits. A false frame mask suppresses that frame, including
// the padded last row of the released JFK processor fixture. With threshold
// 0.5 this is equivalent to a strictly positive finite logit.
func ExtractSegments(logits []float32, frameMask []bool) ([]Segment, error) {
	if len(logits) == 0 || len(logits)%diarizationSpeakers != 0 || len(frameMask) != len(logits)/diarizationSpeakers {
		return nil, fmt.Errorf("invalid Nemotron diarization segment input")
	}
	frames := len(frameMask)
	for _, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization logit")
		}
	}
	segments := make([]Segment, 0)
	for speaker := 0; speaker < diarizationSpeakers; speaker++ {
		start := -1
		for frame := 0; frame <= frames; frame++ {
			active := false
			if frame < frames {
				value := logits[frame*diarizationSpeakers+speaker]
				active = frameMask[frame] && value > 0
			}
			if active && start < 0 {
				start = frame
			}
			if !active && start >= 0 {
				segments = append(segments, Segment{Start: float64(start) / 100, End: float64(frame) / 100, Speaker: speaker})
				start = -1
			}
		}
	}
	sortSegments(segments)
	return segments, nil
}
