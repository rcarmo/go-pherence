package model

import "math"

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
	applyRoPEPartial(x, freqs, pos, numHeads, headDim, headDim/2)
}

// applyRoPEPartial applies RoPE with partial rotation.
// Only the first rotHalf pairs are rotated; remaining dims are untouched.
func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	if pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || len(x) == 0 || len(freqs) == 0 {
		return
	}
	if rotHalf > headDim/2 {
		rotHalf = headDim / 2
	}
	maxHeads := len(x) / headDim
	if numHeads > maxHeads {
		numHeads = maxHeads
	}
	for h := 0; h < numHeads; h++ {
		for i := 0; i < rotHalf; i++ {
			freqOff := (pos*rotHalf + i) * 2
			if freqOff+1 >= len(freqs) {
				break
			}
			cos := freqs[freqOff]
			sin := freqs[freqOff+1]
			idx0 := h*headDim + i
			idx1 := h*headDim + i + rotHalf
			x0 := x[idx0]
			x1 := x[idx1]
			x[idx0] = x0*cos - x1*sin
			x[idx1] = x0*sin + x1*cos
		}
	}
}
