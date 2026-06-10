package llama

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
)

// ApplyRoPE applies full-half rotary position embedding in-place.
func ApplyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	simd.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	llmops.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
