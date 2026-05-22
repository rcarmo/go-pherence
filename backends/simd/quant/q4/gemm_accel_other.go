//go:build !riscv64

package q4

func gemmSymAccelerated(out, x []float32, batch int, qweight, gIdx []int32, scales []float32, inDim, outDim int) bool {
	for b := 0; b < batch; b++ {
		gemvSymScalar(out[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, gIdx, scales, inDim, outDim)
	}
	return true
}
