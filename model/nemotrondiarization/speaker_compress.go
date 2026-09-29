package nemotrondiarization

import (
	"fmt"
	"math"
	"sort"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// SpeakerCompressor owns the released silence embedding used when score-based
// selection leaves empty cache slots. This is a prepared-input operator.
type SpeakerCompressor struct{ silence []float32 }

func LoadSpeakerCompressor(file *safetensors.File) (*SpeakerCompressor, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	values, shape, err := file.GetFloat32("silence_embeds")
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 || shape[0] != projectedWidth || len(values) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization silence embedding shape")
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization silence embedding")
		}
	}
	return &SpeakerCompressor{silence: values}, nil
}

type scoredSpeakerFrame struct {
	index int
	score float32
}

// Compress accepts [rows,512] embeddings and [rows,8] probabilities. It
// returns owned [264,512] embeddings and [264,8] probabilities ordered by
// speaker then original frame index. It does not update FIFO state.
func (m *SpeakerCompressor) Compress(embeds, probs []float32) ([]float32, []float32, error) {
	if m == nil || len(m.silence) != projectedWidth || len(embeds)%projectedWidth != 0 {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization compression model")
	}
	rows := len(embeds) / projectedWidth
	if rows <= diarizationStreamFIFO || rows > 528 || len(probs) != rows*diarizationSpeakers {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization compression input")
	}
	for _, value := range embeds {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron diarization cache embeddings")
		}
	}
	scores, err := SpeakerFrameScores(probs)
	if err != nil {
		return nil, nil, err
	}
	for row := diarizationStreamFIFO; row < rows; row++ {
		for speaker := 0; speaker < diarizationSpeakers; speaker++ {
			scores[row*diarizationSpeakers+speaker] += 0.05
		}
	}
	boost := func(count int, amount float32) {
		candidates := make([]scoredSpeakerFrame, 0, len(scores))
		for i, score := range scores {
			candidates = append(candidates, scoredSpeakerFrame{index: i, score: score})
		}
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].score != candidates[j].score {
				return candidates[i].score > candidates[j].score
			}
			return candidates[i].index < candidates[j].index
		})
		for _, candidate := range candidates[:count] {
			scores[candidate.index] += amount
		}
	}
	boost(24, float32(-2*math.Log(0.5)))
	boost(48, float32(-math.Log(0.5)))
	// Torch flattens speaker-major [8,rows+1] scores. The final slot for
	// each speaker is +Inf and selects the learned silence embedding.
	type selected struct {
		flat  int
		score float32
	}
	flat := make([]selected, 0, (rows+1)*diarizationSpeakers)
	for speaker := 0; speaker < diarizationSpeakers; speaker++ {
		for row := 0; row <= rows; row++ {
			score := float32(math.Inf(1))
			if row < rows {
				score = scores[row*diarizationSpeakers+speaker]
			}
			flat = append(flat, selected{flat: speaker*(rows+1) + row, score: score})
		}
	}
	sort.Slice(flat, func(i, j int) bool {
		if flat[i].score != flat[j].score {
			return flat[i].score > flat[j].score
		}
		return flat[i].flat < flat[j].flat
	})
	flat = flat[:diarizationStreamFIFO]
	for i := range flat {
		if math.IsInf(float64(flat[i].score), -1) {
			flat[i].flat = (rows + 1) * diarizationSpeakers
		}
	}
	sort.Slice(flat, func(i, j int) bool { return flat[i].flat < flat[j].flat })
	out := make([]float32, diarizationStreamFIFO*projectedWidth)
	outProbs := make([]float32, diarizationStreamFIFO*diarizationSpeakers)
	for i, item := range flat {
		row := rows
		if item.flat != (rows+1)*diarizationSpeakers {
			row = min(item.flat%(rows+1), rows)
		}
		if row == rows {
			copy(out[i*projectedWidth:(i+1)*projectedWidth], m.silence)
			continue
		}
		copy(out[i*projectedWidth:(i+1)*projectedWidth], embeds[row*projectedWidth:(row+1)*projectedWidth])
		copy(outProbs[i*diarizationSpeakers:(i+1)*diarizationSpeakers], probs[row*diarizationSpeakers:(row+1)*diarizationSpeakers])
	}
	return out, outProbs, nil
}
