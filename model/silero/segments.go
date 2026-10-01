package silero

import (
	"fmt"
)

// Span is an original mono16-kHz sample interval [Start,End). VAD provides
// speech eligibility only, not word/segment alignment or speaker identity.
type Span struct{ Start, End int64 }

// SegmentOptions applies whisper.cpp's default probability hysteresis and
// duration/padding policy. No finite maximum-speech split is provided yet.
// Limits are checked before arithmetic. Input scores are one per512-sample
// window; any final padded frame is clipped to the actual sample count.
type SegmentOptions struct {
	Threshold                                           float32
	MinSpeechSamples, MinSilenceSamples, PaddingSamples int64
}

func DefaultSegmentOptions() SegmentOptions {
	return SegmentOptions{Threshold: 0.5, MinSpeechSamples: 4000, MinSilenceSamples: 1600, PaddingSamples: 480}
}

// Validate rejects malformed settings before any PCM read/inference.
func (opts SegmentOptions) Validate() error {
	if !finite(opts.Threshold) || opts.Threshold <= 0 || opts.Threshold >= 1 || opts.MinSpeechSamples < 0 || opts.MinSpeechSamples > 16000*60 || opts.MinSilenceSamples < 1 || opts.MinSilenceSamples > 16000*60 || opts.PaddingSamples < 0 || opts.PaddingSamples > 16000*60 {
		return fmt.Errorf("Silero: invalid segmentation options")
	}
	return nil
}

func SpeechSpans(probabilities []float32, totalSamples int64, opts SegmentOptions) ([]Span, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if totalSamples < 1 || totalSamples > 4*3600*16000 || int64(len(probabilities)) != (totalSamples+511)/512 {
		return nil, fmt.Errorf("Silero: invalid segmentation inputs")
	}
	for _, p := range probabilities {
		if !finite(p) || p < 0 || p > 1 {
			return nil, fmt.Errorf("Silero: invalid probability")
		}
	}
	endThreshold := max(opts.Threshold-0.15, float32(0.01))
	spans := make([]Span, 0)
	triggered := false
	var start, tempEnd int64
	for i, p := range probabilities {
		current := int64(i) * 512
		if p >= opts.Threshold && tempEnd != 0 {
			tempEnd = 0
		}
		if p >= opts.Threshold && !triggered {
			triggered = true
			start = current
			continue
		}
		if p < endThreshold && triggered {
			if tempEnd == 0 {
				tempEnd = current
			}
			if current-tempEnd < opts.MinSilenceSamples {
				continue
			}
			if tempEnd-start > opts.MinSpeechSamples {
				spans = append(spans, Span{start, tempEnd})
			}
			triggered = false
			tempEnd = 0
		}
	}
	// The reference's minimum-duration decision uses the padded frame length.
	// Preserve that decision, but bound retained audio to actual PCM: no invented
	// zero-padded audio is exposed by this original-sample API.
	if triggered && int64(len(probabilities))*512-start > opts.MinSpeechSamples {
		spans = append(spans, Span{start, totalSamples})
	}
	// The selected reference merges gaps below200ms before applying padding.
	merged := spans[:0]
	for _, span := range spans {
		if len(merged) > 0 && span.Start-merged[len(merged)-1].End < 3200 {
			merged[len(merged)-1].End = span.End
		} else {
			merged = append(merged, span)
		}
	}
	for i := range merged {
		if i == 0 {
			merged[i].Start = max(int64(0), merged[i].Start-opts.PaddingSamples)
		}
		if i+1 < len(merged) {
			gap := merged[i+1].Start - merged[i].End
			if gap < 2*opts.PaddingSamples {
				merged[i].End += gap / 2
				merged[i+1].Start -= gap / 2
			} else {
				merged[i].End = min(totalSamples, merged[i].End+opts.PaddingSamples)
				merged[i+1].Start = max(int64(0), merged[i+1].Start-opts.PaddingSamples)
			}
		} else {
			merged[i].End = min(totalSamples, merged[i].End+opts.PaddingSamples)
		}
	}
	return merged, nil
}

// TimeMap stores disjoint original speech spans and their concatenated offsets.
// Start/End mapping are intentionally separate: a boundary between retained
// spans maps to the next span for a start and the previous span for an end.
// It cannot map word boundaries inside a removed gap; callers must explicitly
// split cues spanning retained spans and must never invent silence words.
type TimeMap struct {
	spans   []Span
	offsets []int64
	total   int64
}

func NewTimeMap(spans []Span, totalSamples int64) (*TimeMap, error) {
	if totalSamples < 1 || totalSamples > 4*3600*16000 {
		return nil, fmt.Errorf("Silero: invalid timeline")
	}
	m := &TimeMap{spans: append([]Span(nil), spans...), offsets: make([]int64, len(spans))}
	var last int64
	for i, s := range spans {
		if s.Start < last || s.Start < 0 || s.End <= s.Start || s.End > totalSamples {
			return nil, fmt.Errorf("Silero: invalid/overlapping speech span")
		}
		m.offsets[i] = m.total
		m.total += s.End - s.Start
		last = s.End
	}
	return m, nil
}
func (m *TimeMap) Samples() int64 {
	if m == nil {
		return 0
	}
	return m.total
}
func (m *TimeMap) MapStart(sample int64) (int64, error) { return m.mapBoundary(sample, false) }
func (m *TimeMap) MapEnd(sample int64) (int64, error)   { return m.mapBoundary(sample, true) }
func (m *TimeMap) mapBoundary(sample int64, end bool) (int64, error) {
	if m == nil || len(m.spans) == 0 || sample < 0 || sample > m.total {
		return 0, fmt.Errorf("Silero: mapped sample outside speech")
	}
	// binary search for interval ending strictly after sample; end-boundary
	// mapping uses an inclusive interval end to retain the preceding span.
	left, right := 0, len(m.spans)
	for left < right {
		mid := (left + right) / 2
		stop := m.offsets[mid] + m.spans[mid].End - m.spans[mid].Start
		if stop < sample || (!end && stop == sample) {
			left = mid + 1
		} else {
			right = mid
		}
	}
	if left == len(m.spans) {
		return m.spans[len(m.spans)-1].End, nil
	}
	return m.spans[left].Start + sample - m.offsets[left], nil
}
