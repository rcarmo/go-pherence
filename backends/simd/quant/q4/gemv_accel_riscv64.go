//go:build riscv64

package q4

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

func gemvSymAccelerated(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	wrow := make([]float32, inDim)
	for j := 0; j < outDim; j++ {
		dequantSymOutputRow(wrow, qweight, gIdx, scales, inDim, outDim, j)
		out[j] = simd.Sdot(x[:inDim], wrow)
	}
}

func dequantSymOutputRow(out []float32, qweight, gIdx []int32, scales []float32, inDim, outDim, j int) {
	for packIdx := 0; packIdx < inDim/8; packIdx++ {
		qw := qweight[packIdx*outDim+j]
		for bit := 0; bit < 8; bit++ {
			i := packIdx*8 + bit
			g := int(gIdx[i])
			qv := (qw >> (uint(bit) * 4)) & 0xF
			out[i] = scales[g*outDim+j] * float32(qv-8)
		}
	}
}
