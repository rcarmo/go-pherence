package qwen

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

// qwen35BranchConvRow writes a fixed 6144-channel four-tap causal convolution.
// All inputs are validated internal scratch; dst and products are distinct and
// disjoint from x/weights. Separate SIMD multiply/add keeps the reference's two
// F32 roundings per tap, unlike Saxpy/FMA. No request state persists here.
func qwen35BranchConvRow(dst, x, weights, products []float32, parents []int, token int) {
	clear(dst)
	for tap := 0; tap < 4; tap++ {
		p := token
		for back := 0; back < 3-tap && p >= 0; back++ {
			p = parents[p]
		}
		if p >= 0 {
			src, weight := x[p*6144:(p+1)*6144], weights[tap*6144:(tap+1)*6144]
			if simd.HasVecAsm {
				simd.VecMul(products, src, weight)
				simd.VecAdd(dst, dst, products)
			} else {
				// Avoid a second memory pass on scalar-only targets.
				for c := range dst {
					dst[c] += float32(src[c] * weight[c])
				}
			}
		}
	}
}
