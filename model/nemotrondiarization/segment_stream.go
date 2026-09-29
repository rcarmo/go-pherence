package nemotrondiarization

import (
	"fmt"
	"math"
	"sort"
)

// SegmentStream converts committed [frames,8] logits into completed speaker
// spans. It retains one open start per speaker, independent of recording
// length. Returned segments are owned. A stream must not be used concurrently.
type SegmentStream struct {
	starts [diarizationSpeakers]int64
	frames int64
	closed bool
}

// Append consumes complete, finite 10-ms logits. Validation failures do not
// advance state. A span touching the end of this call is kept open until a
// later inactive frame or Finish. The caller must consume returned spans.
func (s *SegmentStream) Append(logits []float32) ([]Segment, error) {
	if s == nil || s.closed || len(logits) == 0 || len(logits)%diarizationSpeakers != 0 || s.frames > math.MaxInt64-int64(len(logits)/diarizationSpeakers) {
		return nil, fmt.Errorf("invalid Nemotron diarization segment stream chunk")
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization segment stream logit")
		}
	}
	out := make([]Segment, 0)
	for row := 0; row < len(logits)/diarizationSpeakers; row++ {
		frame := s.frames + int64(row)
		for speaker := range s.starts {
			active := logits[row*diarizationSpeakers+speaker] > 0
			start := s.starts[speaker]
			if active && start == 0 {
				s.starts[speaker] = frame + 1 // offset permits an open span at frame zero
			} else if !active && start != 0 {
				out = append(out, Segment{Start: float64(start-1) / 100, End: float64(frame) / 100, Speaker: speaker})
				s.starts[speaker] = 0
			}
		}
	}
	s.frames += int64(len(logits) / diarizationSpeakers)
	sortSegments(out)
	return out, nil
}

// Finish closes all open spans at the last committed frame. It can be called
// once after at least one logits row; an empty stream cannot be finished.
func (s *SegmentStream) Finish() ([]Segment, error) {
	if s == nil || s.closed || s.frames == 0 {
		return nil, fmt.Errorf("invalid Nemotron diarization segment stream finish")
	}
	out := make([]Segment, 0, diarizationSpeakers)
	for speaker, start := range s.starts {
		if start != 0 {
			out = append(out, Segment{Start: float64(start-1) / 100, End: float64(s.frames) / 100, Speaker: speaker})
			s.starts[speaker] = 0
		}
	}
	s.closed = true
	sortSegments(out)
	return out, nil
}

func sortSegments(segments []Segment) {
	sort.Slice(segments, func(i, j int) bool {
		if segments[i].Start != segments[j].Start {
			return segments[i].Start < segments[j].Start
		}
		return segments[i].Speaker < segments[j].Speaker
	})
}
