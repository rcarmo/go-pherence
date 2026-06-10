package ops

import "github.com/rcarmo/go-pherence/backends/simd/runtime"

// GemvNT computes out = x @ w^T where w is [outDim, inDim].
func GemvNT(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	if len(out) < outDim {
		return
	}
	simd.GemvRows(out[:outDim], x, w, outDim, inDim)
}

func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	simd.ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, rotHalf)
}
