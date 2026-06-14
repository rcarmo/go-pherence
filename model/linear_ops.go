package model

import (
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
)

func rmsNormInPlace(x, weight []float32, eps float32) {
	simd.RMSNorm(x, weight, eps)
}

// gemv: out = x @ w where w is either:
//
//	pre-transposed [inDim, outDim] (use NN), or
//	original [outDim, inDim] (use NT via dot products)
func gemv(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	if len(out) < outDim {
		return
	}
	// Detect layout: if w is [inDim, outDim] (pre-transposed), use NN.
	// If w is [outDim, inDim] (original), use NT (dot per output).
	// Heuristic: try NN first (pre-transposed path).
	simd.SgemmNNTo(out[:outDim], x, w, 1, outDim, inDim, 1.0, inDim, outDim, outDim)
}

// gemvNT: out = x @ w^T where w is [outDim, inDim] (original layout)
func gemvNT(out, x []float32, w []float32, inDim, outDim int) {
	llmops.GemvNT(out, x, w, inDim, outDim)
}

func geluTanh(x float32) float32 {
	return simd.GELUTanhScalar(x)
}

// gemvNTParallel is like gemvNT but parallelized across CPU cores.
func gemvNTParallel(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	if len(out) < outDim {
		return
	}
	simd.GemvRowsParallel(out[:outDim], x, w, outDim, inDim)
}
