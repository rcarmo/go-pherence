package simd

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

const hasRoPEAsm = false

func applyRoPEGo(x, freqs []float32, pos, numHeads, headDim int) {
	kernels.ApplyRoPE(x, freqs, pos, numHeads, headDim)
}

func applyRoPEPartialGo(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	kernels.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}

// ApplyRoPE applies full-half rotary position embedding in-place.
func ApplyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	// Dispatch hook kept explicit so AVX2/NEON kernels can be wired without
	// changing callers. Until hasRoPEAsm flips true, the scalar kernel is the
	// reference implementation and public runtime path.
	applyRoPEGo(x, freqs, pos, numHeads, headDim)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	// See ApplyRoPE: scalar remains the only active path until architecture
	// kernels land and pass parity gates.
	applyRoPEPartialGo(x, freqs, pos, numHeads, headDim, rotHalf)
}
