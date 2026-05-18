package model

import (
	"math"

	llamaops "github.com/rcarmo/go-pherence/model/llama"
)

func (m *LlamaModel) precomputeRoPE() {
	cfg := m.Config
	headDim := cfg.HeadDim
	if headDim == 0 {
		headDim = cfg.HiddenSize / cfg.NumHeads
	}
	// For models with variable head_dim (Gemma4), use the max
	if cfg.GlobalHeadDim > headDim {
		headDim = cfg.GlobalHeadDim
	}
	halfDim := headDim / 2
	maxSeq := cfg.MaxSeqLen
	if maxSeq > 2048 {
		maxSeq = 2048 // cap for memory
	}

	m.RopeFreqs = make([]float32, maxSeq*halfDim*2)
	theta := cfg.RopeTheta
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			off := (pos*halfDim + i) * 2
			m.RopeFreqs[off] = float32(math.Cos(angle))
			m.RopeFreqs[off+1] = float32(math.Sin(angle))
		}
	}
}

func applyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	llamaops.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	llamaops.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
