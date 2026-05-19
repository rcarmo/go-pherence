package quant

import simdq4 "github.com/rcarmo/go-pherence/backends/simd/runtime/q4"

// ValidateGemvQ4Sym checks inputs for GemvQ4Sym.
func ValidateGemvQ4Sym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) error {
	return simdq4.ValidateGemvSym(out, x, qweight, gIdx, scales, inDim, outDim)
}
