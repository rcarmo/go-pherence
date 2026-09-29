package nemotrondiarization

import (
	"fmt"
	"math"
)

// PoolSpeakerProbabilities applies the released streaming cache's sigmoid
// and eight-logit average pool, then zeros masked encoder rows. The returned
// [rows,8] slice is owned. It does not score or compress speaker frames.
func PoolSpeakerProbabilities(logits []float32, mask []bool) ([]float32, error) {
	rows := len(mask)
	if rows < 1 || rows > 376 || len(logits) != rows*diarizationUpsample*diarizationSpeakers {
		return nil, fmt.Errorf("invalid Nemotron diarization pooled logits")
	}
	for _, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization pooled logits")
		}
	}
	out := make([]float32, rows*diarizationSpeakers)
	for row := 0; row < rows; row++ {
		if !mask[row] {
			continue
		}
		for speaker := 0; speaker < diarizationSpeakers; speaker++ {
			var sum float32
			for frame := 0; frame < diarizationUpsample; frame++ {
				value := float64(logits[(row*diarizationUpsample+frame)*diarizationSpeakers+speaker])
				sum += float32(1 / (1 + math.Exp(-value)))
			}
			out[row*diarizationSpeakers+speaker] = sum / diarizationUpsample
		}
	}
	return out, nil
}
