package rope

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

// ApplyRoPETo applies full-half RoPE and reports malformed inputs.
func ApplyRoPETo(x, freqs []float32, pos, numHeads, headDim int) bool {
	return ApplyRoPEPartialTo(x, freqs, pos, numHeads, headDim, headDim/2)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	// See ApplyRoPE: scalar remains the only active path until architecture
	// kernels land and pass parity gates.
	applyRoPEPartialGo(x, freqs, pos, numHeads, headDim, rotHalf)
}

// ApplyRoPEPartialTo applies partial RoPE and reports malformed inputs.
func ApplyRoPEPartialTo(x, freqs []float32, pos, numHeads, headDim, rotHalf int) bool {
	if pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 {
		return false
	}
	total, okTotal := checkedMulInt(numHeads, headDim)
	posPairs, okPos := checkedMulInt(pos+1, rotHalf)
	freqNeed, okFreq := checkedMulInt(posPairs, 2)
	if !okTotal || !okPos || !okFreq || len(x) < total || len(freqs) < freqNeed {
		return false
	}
	applyRoPEPartialGo(x, freqs, pos, numHeads, headDim, rotHalf)
	return true
}
