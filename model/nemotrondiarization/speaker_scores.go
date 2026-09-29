package nemotrondiarization

import (
	"fmt"
	"math"
)

// SpeakerFrameScores implements the released cache's per-speaker frame score
// before recency boosts and top-k compression. Owned output is [rows,8] and
// uses -Inf for rejected frames; input probabilities must be finite [0,1].
func SpeakerFrameScores(probs []float32) ([]float32, error) {
	if len(probs) == 0 || len(probs)%diarizationSpeakers != 0 || len(probs)/diarizationSpeakers > 528 {
		return nil, fmt.Errorf("invalid Nemotron diarization score shape")
	}
	rows := len(probs) / diarizationSpeakers
	out := make([]float32, len(probs))
	var positives [diarizationSpeakers]int
	for row := 0; row < rows; row++ {
		var complements float32
		for speaker := 0; speaker < diarizationSpeakers; speaker++ {
			p := probs[row*diarizationSpeakers+speaker]
			if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || p < 0 || p > 1 {
				return nil, fmt.Errorf("invalid Nemotron diarization probability")
			}
			complements += float32(math.Log(float64(max(1-p, 0.25))))
		}
		for speaker := 0; speaker < diarizationSpeakers; speaker++ {
			p := probs[row*diarizationSpeakers+speaker]
			if p <= 0.5 {
				out[row*diarizationSpeakers+speaker] = float32(math.Inf(-1))
				continue
			}
			complement := float32(math.Log(float64(max(1-p, 0.25))))
			score := float32(math.Log(float64(max(p, 0.25)))) - complement + complements - float32(math.Log(0.5))
			out[row*diarizationSpeakers+speaker] = score
			if score > 0 {
				positives[speaker]++
			}
		}
	}
	for speaker, count := range positives {
		if count < 16 {
			continue
		}
		for row := 0; row < rows; row++ {
			i := row*diarizationSpeakers + speaker
			if out[i] <= 0 {
				out[i] = float32(math.Inf(-1))
			}
		}
	}
	return out, nil
}
