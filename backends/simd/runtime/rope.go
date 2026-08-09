package simd

import (
	"github.com/rcarmo/go-pherence/backends/simd/kernels"
	"github.com/rcarmo/go-pherence/internal/checked"
)

func applyRoPEGo(x, freqs []float32, pos, numHeads, headDim int) {
	kernels.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

func applyRoPEPartialGo(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	kernels.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}

// ApplyRoPE applies full-half rotary position embedding in-place.
func ApplyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, headDim/2)
}

// ApplyRoPETo applies full-half RoPE and reports malformed inputs.
func ApplyRoPETo(x, freqs []float32, pos, numHeads, headDim int) bool {
	return ApplyRoPEPartialTo(x, freqs, pos, numHeads, headDim, headDim/2)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	if applyRoPEPartialAccel(x, freqs, pos, numHeads, headDim, rotHalf) {
		return
	}
	applyRoPEPartialGo(x, freqs, pos, numHeads, headDim, rotHalf)
}

// ApplyRoPEPartialTo applies partial RoPE and reports malformed inputs.
func ApplyRoPEPartialTo(x, freqs []float32, pos, numHeads, headDim, rotHalf int) bool {
	if pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 {
		return false
	}
	total, okTotal := checked.MulInt(numHeads, headDim)
	posPairs, okPos := checked.MulInt(pos+1, rotHalf)
	freqNeed, okFreq := checked.MulInt(posPairs, 2)
	if !okTotal || !okPos || !okFreq || len(x) < total || len(freqs) < freqNeed {
		return false
	}
	if !applyRoPEPartialAccel(x, freqs, pos, numHeads, headDim, rotHalf) {
		applyRoPEPartialGo(x, freqs, pos, numHeads, headDim, rotHalf)
	}
	return true
}
