package simd

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

// ApplyRoPE applies full-half rotary position embedding in-place.
func ApplyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	kernels.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	kernels.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
