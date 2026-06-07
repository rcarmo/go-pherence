package k3engine

import (
	"math"
	"math/rand"
)

// Sampler implements temperature-scaled sampling with repetition penalty,
// matching llama.cpp sampler_chain logic.
type Sampler struct {
	Temperature float32
	RepPenalty  float32
	RepWindow   int
	TopK        int
	EOSTokens   []int32
	history     []int32
	rng         *rand.Rand
}

func NewSampler(temp, repPenalty float32, repWindow, topK int, eosTokens []int32) *Sampler {
	return &Sampler{
		Temperature: temp,
		RepPenalty:  repPenalty,
		RepWindow:   repWindow,
		TopK:        topK,
		EOSTokens:   eosTokens,
		rng:         rand.New(rand.NewSource(42)),
	}
}

// Sample selects the next token from logits.
// Returns (token_id, is_eos).
func (s *Sampler) Sample(logits []float32) (int32, bool) {
	nVocab := len(logits)

	// 1. Apply repetition penalty
	if s.RepPenalty != 1.0 && len(s.history) > 0 {
		start := 0
		if len(s.history) > s.RepWindow {
			start = len(s.history) - s.RepWindow
		}
		for _, tok := range s.history[start:] {
			if int(tok) < nVocab {
				if logits[tok] > 0 {
					logits[tok] /= s.RepPenalty
				} else {
					logits[tok] *= s.RepPenalty
				}
			}
		}
	}

	// 2. Temperature = 0 → greedy
	if s.Temperature <= 0 {
		return s.argmax(logits)
	}

	// 3. Apply temperature
	invTemp := 1.0 / s.Temperature
	for i := range logits {
		logits[i] *= invTemp
	}

	// 4. Top-K filtering
	topK := s.TopK
	if topK <= 0 || topK > nVocab {
		topK = nVocab
	}
	// Find top-K indices (partial sort)
	type candidate struct {
		idx int32
		val float32
	}
	topCands := make([]candidate, 0, topK)
	for i := 0; i < nVocab; i++ {
		if len(topCands) < topK {
			topCands = append(topCands, candidate{int32(i), logits[i]})
			// bubble up
			for j := len(topCands) - 1; j > 0 && topCands[j].val > topCands[j-1].val; j-- {
				topCands[j], topCands[j-1] = topCands[j-1], topCands[j]
			}
		} else if logits[i] > topCands[topK-1].val {
			topCands[topK-1] = candidate{int32(i), logits[i]}
			// bubble up
			for j := topK - 1; j > 0 && topCands[j].val > topCands[j-1].val; j-- {
				topCands[j], topCands[j-1] = topCands[j-1], topCands[j]
			}
		}
	}

	// 5. Softmax over top-K
	maxVal := topCands[0].val
	sumExp := float64(0)
	probs := make([]float64, len(topCands))
	for i, c := range topCands {
		p := math.Exp(float64(c.val - maxVal))
		probs[i] = p
		sumExp += p
	}
	for i := range probs {
		probs[i] /= sumExp
	}

	// 6. Random sample from distribution
	r := s.rng.Float64()
	cumProb := float64(0)
	selectedIdx := int32(topCands[0].idx)
	for i, p := range probs {
		cumProb += p
		if r <= cumProb {
			selectedIdx = topCands[i].idx
			break
		}
	}

	// 7. Record history and check EOS
	s.history = append(s.history, selectedIdx)
	for _, eos := range s.EOSTokens {
		if selectedIdx == eos {
			return selectedIdx, true
		}
	}
	return selectedIdx, false
}

// argmax returns the index of the maximum value (greedy with rep penalty applied).
func (s *Sampler) argmax(logits []float32) (int32, bool) {
	maxIdx := int32(0)
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = int32(i)
		}
	}
	s.history = append(s.history, maxIdx)
	for _, eos := range s.EOSTokens {
		if maxIdx == eos {
			return maxIdx, true
		}
	}
	return maxIdx, false
}

// Reset clears the history for a new generation.
func (s *Sampler) Reset() {
	s.history = s.history[:0]
}
