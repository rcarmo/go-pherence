package model

import (
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func (m *LlamaModel) precomputeRoPE() {
	cfg := m.Config
	headDim := cfg.HeadDim
	if headDim == 0 && cfg.NumHeads > 0 {
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

	m.RopeFreqs = buildRoPEFreqs(maxSeq, halfDim, headDim, cfg.RopeTheta)
}

func buildRoPEFreqs(maxSeq, halfDim, headDim int, theta float64) []float32 {
	n, ok := checkedRoPEFreqLen(maxSeq, halfDim)
	if !ok || headDim <= 0 {
		return nil
	}
	if theta <= 0 {
		theta = 10000
	}
	freqs := make([]float32, n)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(angle))
			freqs[off+1] = float32(math.Sin(angle))
		}
	}
	return freqs
}

func checkedRoPEFreqLen(maxSeq, halfDim int) (int, bool) {
	if maxSeq <= 0 || halfDim <= 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if maxSeq > maxInt/halfDim {
		return 0, false
	}
	n := maxSeq * halfDim
	if n > maxInt/2 {
		return 0, false
	}
	return n * 2, true
}

func applyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	simd.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	simd.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
