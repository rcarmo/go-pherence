package qwen3tts

import (
	"fmt"
	"math"
	"slices"
)

// ReferenceCPUSampler follows the pinned Rust/Candle CPU semantic-token
// sampler on the qualified fixture rows. It owns its PCG state; callers must
// not share a sampler concurrently. This is not a synthesis API.
type ReferenceCPUSampler struct{ state uint64 }

func NewReferenceCPUSampler(seed uint64) *ReferenceCPUSampler {
	return &ReferenceCPUSampler{state: seed*2685821657736338717 + 1442695040888963407}
}

func (s *ReferenceCPUSampler) draw() float32 {
	old := s.state
	s.state = old*6364136223846793005 + 1442695040888963407
	x := uint32(((old >> 18) ^ old) >> 27)
	rot := int(old >> 59)
	return float32((x>>rot)|(x<<((-rot)&31))) / float32(math.MaxUint32)
}

// ReferenceSampleConfig selects the pinned CPU temperature/top-k/top-p policy.
// Repetition and TTS control suppression are explicit inputs to each call.
type ReferenceSampleConfig struct {
	Temperature       float64
	TopK              int
	TopP              float64
	RepetitionPenalty float64
}

// Select copies logits before applying sign-aware repetition penalty, control
// suppression and optional EOS minimum. It returns an owned semantic ID.
func (s *ReferenceCPUSampler) Select(logits []float32, cfg ReferenceSampleConfig, seen []uint32, suppressControls bool, allowEOS bool) (uint32, error) {
	if s == nil || len(logits) == 0 || cfg.TopK < 0 || math.IsNaN(cfg.Temperature) || math.IsInf(cfg.Temperature, 0) || cfg.Temperature < 0 || (cfg.Temperature > 0 && cfg.Temperature < math.SmallestNonzeroFloat32) || math.IsNaN(cfg.TopP) || cfg.TopP < 0 || cfg.TopP > 1 || math.IsNaN(cfg.RepetitionPenalty) || math.IsInf(cfg.RepetitionPenalty, 0) || cfg.RepetitionPenalty < math.SmallestNonzeroFloat32 || cfg.RepetitionPenalty > math.MaxFloat32 {
		return 0, fmt.Errorf("invalid Qwen3-TTS reference sampling configuration")
	}
	if suppressControls && len(logits) != CodecVocabSize {
		return 0, fmt.Errorf("invalid Qwen3-TTS sampling vocab=%d", len(logits))
	}
	row := append([]float32(nil), logits...)
	for _, v := range row {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 1) {
			return 0, fmt.Errorf("nonfinite Qwen3-TTS sampling logit")
		}
	}
	if cfg.RepetitionPenalty != 1 && len(seen) > 0 {
		visited := make([]bool, len(row))
		penalty := float32(cfg.RepetitionPenalty)
		for _, id := range seen {
			if int(id) < len(row) {
				visited[id] = true
			}
		}
		for i, yes := range visited {
			if yes {
				if row[i] > 0 {
					row[i] /= penalty
				} else {
					row[i] *= penalty
				}
				if math.IsNaN(float64(row[i])) || math.IsInf(float64(row[i]), 0) {
					return 0, fmt.Errorf("nonfinite Qwen3-TTS penalized sampling logit")
				}
			}
		}
	}
	if suppressControls {
		for i := len(row) - 1024; i < len(row); i++ {
			if i != int(CodecEOS) {
				row[i] = float32(math.Inf(-1))
			}
		}
	}
	if !allowEOS && int(CodecEOS) < len(row) {
		row[CodecEOS] = float32(math.Inf(-1))
	}
	if cfg.Temperature < 0.01 {
		best := -1
		for i, v := range row {
			if !math.IsInf(float64(v), -1) && (best < 0 || v > row[best]) {
				best = i
			}
		}
		if best < 0 {
			return 0, fmt.Errorf("no Qwen3-TTS sampling candidate")
		}
		return uint32(best), nil
	}
	for i, v := range row {
		if !math.IsInf(float64(v), -1) {
			row[i] = v / float32(cfg.Temperature)
			if math.IsNaN(float64(row[i])) || math.IsInf(float64(row[i]), 0) {
				return 0, fmt.Errorf("nonfinite Qwen3-TTS scaled sampling logit")
			}
		}
	}
	if cfg.TopK > 0 && cfg.TopK < len(row) {
		sorted := slices.Clone(row)
		slices.SortFunc(sorted, func(a, b float32) int {
			if a > b {
				return -1
			}
			if a < b {
				return 1
			}
			return 0
		})
		threshold := sorted[cfg.TopK-1]
		for i, v := range row {
			if v < threshold {
				row[i] = float32(math.Inf(-1))
			}
		}
	}
	if cfg.TopP > 0 && cfg.TopP < 1 {
		indices := make([]int, len(row))
		for i := range indices {
			indices[i] = i
		}
		slices.SortFunc(indices, func(a, b int) int {
			if row[a] > row[b] {
				return -1
			}
			if row[a] < row[b] {
				return 1
			}
			return 0
		})
		maxValue := row[indices[0]]
		probabilities := make([]float32, len(row))
		var sum float32
		for i, id := range indices {
			p := float32(math.Exp(float64(row[id] - maxValue)))
			probabilities[i] = p
			sum += p
		}
		if sum == 0 {
			return 0, fmt.Errorf("no Qwen3-TTS sampling probability")
		}
		var cumulative float32
		cut := len(row)
		for i, p := range probabilities {
			cumulative += p / sum
			if cumulative > float32(cfg.TopP) {
				cut = i + 1
				break
			}
		}
		for _, id := range indices[cut:] {
			row[id] = float32(math.Inf(-1))
		}
	}
	maxValue := float32(math.Inf(-1))
	for _, v := range row {
		if v > maxValue {
			maxValue = v
		}
	}
	if math.IsInf(float64(maxValue), -1) {
		return 0, fmt.Errorf("no Qwen3-TTS sampling candidate")
	}
	probs := make([]float32, len(row))
	var sum float32
	for i, v := range row {
		p := float32(math.Exp(float64(v - maxValue)))
		probs[i] = p
		sum += p
	}
	if sum == 0 {
		return 0, fmt.Errorf("no Qwen3-TTS sampling probability")
	}
	draw := s.draw()
	var cumulative float32
	for i, p := range probs {
		cumulative += p / sum
		if cumulative >= draw {
			return uint32(i), nil
		}
	}
	return uint32(len(row) - 1), nil
}
