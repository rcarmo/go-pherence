//go:build riscv64

package q4

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

func gemmSymAccelerated(out, x []float32, batch int, qweight, gIdx []int32, scales []float32, inDim, outDim int) bool {
	return gemmSymBatchedPortable(out, x, batch, qweight, gIdx, scales, inDim, outDim)
}

func gemmAccelerated(out, x []float32, batch int, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int) bool {
	wrow := make([]float32, inDim)
	for j := 0; j < outDim; j++ {
		dequantOutputRow(wrow, qweight, qzeros, gIdx, scales, inDim, outDim, j)
		for b := 0; b < batch; b++ {
			out[b*outDim+j] = simd.Sdot(x[b*inDim:(b+1)*inDim], wrow)
		}
	}
	return true
}
