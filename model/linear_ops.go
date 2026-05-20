package model

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
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
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	if len(w) >= weightLen {
		// Detect layout: if w is [inDim, outDim] (pre-transposed), use NN
		// If w is [outDim, inDim] (original), use NT (dot per output)
		// Heuristic: try NN first (pre-transposed path)
		if simd.HasSgemmAsm {
			simd.SgemmNN(1, outDim, inDim, 1.0,
				unsafe.Pointer(&x[0]), unsafe.Pointer(&w[0]), unsafe.Pointer(&out[0]),
				inDim, outDim, outDim)
		} else {
			simd.GemvCols(out[:outDim], x[:inDim], w[:weightLen], inDim, outDim)
		}
	}
}

// gemvNT: out = x @ w^T where w is [outDim, inDim] (original layout)
func gemvNT(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	simd.GemvRows(out[:outDim], x[:inDim], w[:weightLen], outDim, inDim)
}

func checkedProduct(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func geluTanh(x float32) float32 {
	return simd.GELUTanhScalar(x)
}

// gemvNTParallel is like gemvNT but parallelized across CPU cores.
func gemvNTParallel(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	simd.GemvRows(out[:outDim], x[:inDim], w[:weightLen], outDim, inDim)
}
