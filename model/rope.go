package model

import (
	"math"

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
	m.precomputeGemma4RoPEWithFullFactors(nil)
}

func (m *LlamaModel) precomputeGemma4RoPEWithFullFactors(fullFactors []float32) {
	cfg := m.Config
	if cfg.ModelType != "gemma4_text" {
		return
	}
	m.RopeFreqsSWA, m.RopeHalfSWA, m.RopeFreqsFull, m.RopeHalfFull = gemma4RoPETables(cfg, fullFactors)
}

func (d *Gemma4MTPDrafter) precomputeGemma4RoPEWithFullFactors(fullFactors []float32) {
	if d == nil || d.Config.ModelType != "gemma4_text" {
		return
	}
	d.RopeFreqsSWA, d.RopeHalfSWA, d.RopeFreqsFull, d.RopeHalfFull = gemma4RoPETables(d.Config, fullFactors)
}

func gemma4RoPETables(cfg LlamaConfig, fullFactors []float32) (swa []float32, swaHalf int, full []float32, fullHalf int) {
	maxSeq := cfg.MaxSeqLen
	if maxSeq > 2048 {
		maxSeq = 2048
	}
	if cfg.HeadDim > 0 {
		swaHalf = cfg.HeadDim / 2
		swa = buildGemma4GGMLRoPEFreqs(maxSeq, swaHalf, cfg.HeadDim, 10000)
	}
	if cfg.GlobalHeadDim > 0 {
		// llama.cpp applies Gemma4 full-attention RoPE across the whole global
		// head dimension (n_rot=head_dim) and uses rope_freqs.weight as
		// proportional factors. The first quarter of pairs normally has factor 1;
		// the rest has a very large factor, making them effectively identity while
		// preserving llama.cpp's full-head pairing layout.
		fullHalf = cfg.GlobalHeadDim / 2
		if len(fullFactors) == 0 {
			fullFactors = synthesizedGemma4FullAttentionRoPEFactors(cfg.GlobalHeadDim)
		}
		full = buildGemma4GGMLRoPEFreqsWithFactors(maxSeq, fullHalf, cfg.GlobalHeadDim, 1000000, fullFactors)
	}
	return swa, swaHalf, full, fullHalf
}

func synthesizedGemma4FullAttentionRoPEFactors(headDim int) []float32 {
	half := headDim / 2
	if half <= 0 {
		return nil
	}
	factors := make([]float32, half)
	enabledPairs := int(float64(headDim)*0.25) / 2
	if enabledPairs < 0 {
		enabledPairs = 0
	}
	if enabledPairs > half {
		enabledPairs = half
	}
	for i := range factors {
		if i < enabledPairs {
			factors[i] = 1
		} else {
			factors[i] = 1e30
		}
	}
	return factors
}

func buildGemma4GGMLRoPEFreqs(maxSeq, halfDim, headDim int, theta float64) []float32 {
	n, ok := simd.CheckedRoPEFreqLen(maxSeq, halfDim)
	if !ok || headDim <= 0 {
		return nil
	}
	if theta <= 0 {
		theta = 10000
	}
	freqs := make([]float32, n)
	thetaScale := float32(math.Pow(theta, -2.0/float64(headDim)))
	for pos := 0; pos < maxSeq; pos++ {
		thetaCur := float32(pos)
		for i := 0; i < halfDim; i++ {
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(float64(thetaCur)))
			freqs[off+1] = float32(math.Sin(float64(thetaCur)))
			thetaCur *= thetaScale
		}
	}
	return freqs
}

func buildGemma4GGMLRoPEFreqsWithFactors(maxSeq, halfDim, nDims int, theta float64, factors []float32) []float32 {
	n, ok := simd.CheckedRoPEFreqLen(maxSeq, halfDim)
	if !ok || nDims <= 0 {
		return nil
	}
	if theta <= 0 {
		theta = 10000
	}
	freqs := make([]float32, n)
	thetaScale := float32(math.Pow(theta, -2.0/float64(nDims)))
	for pos := 0; pos < maxSeq; pos++ {
		thetaCur := float32(pos)
		for i := 0; i < halfDim; i++ {
			factor := float32(1)
			if i < len(factors) && factors[i] != 0 {
				factor = factors[i]
			}
			angle := thetaCur / factor
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(float64(angle)))
			freqs[off+1] = float32(math.Sin(float64(angle)))
			thetaCur *= thetaScale
		}
	}
	return freqs
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
