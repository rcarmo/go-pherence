//go:build !riscv64

package q4

func gemvSymAccelerated(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	gemvSymScalar(out, x, qweight, gIdx, scales, inDim, outDim)
}

func gemvAccelerated(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int) {
	gemvScalar(out, x, qweight, qzeros, gIdx, scales, inDim, outDim)
}
