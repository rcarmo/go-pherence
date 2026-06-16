package model

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
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

func (m *LlamaModel) precomputeGemma4RoPE() {
	cfg := m.Config
	if cfg.ModelType != "gemma4_text" {
		return
	}
	maxSeq := cfg.MaxSeqLen
	if maxSeq > 2048 {
		maxSeq = 2048
	}
	if cfg.HeadDim > 0 {
		m.RopeHalfSWA = cfg.HeadDim / 2
		m.RopeFreqsSWA = buildRoPEFreqs(maxSeq, m.RopeHalfSWA, cfg.HeadDim, 10000)
	}
	if cfg.GlobalHeadDim > 0 {
		rotatedDims := int(float64(cfg.GlobalHeadDim) * 0.25)
		m.RopeHalfFull = rotatedDims / 2
		m.RopeFreqsFull = buildRoPEFreqs(maxSeq, m.RopeHalfFull, cfg.GlobalHeadDim, 1000000)
	}
}

func buildRoPEFreqs(maxSeq, halfDim, headDim int, theta float64) []float32 {
	return simd.BuildRoPEFreqs(maxSeq, halfDim, headDim, theta)
}

func applyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	simd.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	llmops.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
