package speaker

import "github.com/rcarmo/go-pherence/models/whisper"

// DiarizedSegment combines transcription with speaker identity.
type DiarizedSegment struct {
	Start   float64 // seconds
	End     float64 // seconds
	Speaker int     // speaker label (0-indexed)
	Text    string
}

// AlignSpeakers aligns Whisper timestamp segments with speaker diarization labels.
// whisperSegments: timestamped transcript segments from Whisper
// vadSegments: voice-active segments used for speaker embedding
// speakerLabels: cluster labels for each VAD segment (same length as vadSegments)
func AlignSpeakers(whisperSegments []whisper.Segment, vadSegments []VADSegment, speakerLabels []int) []DiarizedSegment {
	if len(whisperSegments) == 0 {
		return nil
	}

	result := make([]DiarizedSegment, 0, len(whisperSegments))
	for _, ws := range whisperSegments {
		speaker := findOverlappingSpeaker(ws.Start, ws.End, vadSegments, speakerLabels)
		result = append(result, DiarizedSegment{
			Start:   ws.Start,
			End:     ws.End,
			Speaker: speaker,
			Text:    ws.Text,
		})
	}
	return result
}

// findOverlappingSpeaker finds which speaker label has the most overlap with [start, end].
func findOverlappingSpeaker(start, end float64, vadSegs []VADSegment, labels []int) int {
	if len(vadSegs) == 0 || len(labels) == 0 {
		return 0
	}

	// Find maximum overlap per speaker
	overlapByLabel := make(map[int]float64)
	for i, seg := range vadSegs {
		if i >= len(labels) {
			break
		}
		overlap := overlapDuration(start, end, seg.Start, seg.End)
		if overlap > 0 {
			overlapByLabel[labels[i]] += overlap
		}
	}

	if len(overlapByLabel) == 0 {
		// No overlap: find nearest segment
		return findNearestSpeaker(start, vadSegs, labels)
	}

	// Return label with most overlap
	bestLabel := 0
	bestOverlap := 0.0
	for label, overlap := range overlapByLabel {
		if overlap > bestOverlap {
			bestOverlap = overlap
			bestLabel = label
		}
	}
	return bestLabel
}

func overlapDuration(s1, e1, s2, e2 float64) float64 {
	start := s1
	if s2 > start {
		start = s2
	}
	end := e1
	if e2 < end {
		end = e2
	}
	if end > start {
		return end - start
	}
	return 0
}

func findNearestSpeaker(t float64, vadSegs []VADSegment, labels []int) int {
	if len(vadSegs) == 0 {
		return 0
	}
	bestIdx := 0
	bestDist := abs64(t - vadSegs[0].Start)
	for i, seg := range vadSegs[1:] {
		d := abs64(t - seg.Start)
		if d < bestDist {
			bestDist = d
			bestIdx = i + 1
		}
	}
	if bestIdx < len(labels) {
		return labels[bestIdx]
	}
	return 0
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
