package quant

import simdq4 "github.com/rcarmo/go-pherence/backends/simd/runtime/q4"

// GemvQ4 computes out = x @ W^T where W is stored as GPTQ INT4.
func GemvQ4(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int, sym bool) {
	simdq4.Gemv(out, x, qweight, qzeros, gIdx, scales, inDim, outDim, sym)
}

// GemvQ4Sym computes out = x @ W^T where W is stored as GPTQ INT4 symmetric.
// This dequantizes on-the-fly during the dot product, avoiding the full F32 expansion.
//
// qweight: [inDim/8, outDim] packed int32
// scales:  [numGroups, outDim] float32
// gIdx:    [inDim] int32
// x:       [inDim] float32
// out:     [outDim] float32
func GemvQ4Sym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	simdq4.GemvSym(out, x, qweight, gIdx, scales, inDim, outDim)
}
