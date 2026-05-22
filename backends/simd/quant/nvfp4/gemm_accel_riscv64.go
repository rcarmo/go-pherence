//go:build riscv64

package nvfp4

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

func gemmNVFP4Accelerated(out, x []float32, batch int, qw *NVFP4Weight) bool {
	wrow := make([]float32, qw.InDim)
	for row := 0; row < qw.OutDim; row++ {
		dequantNVFP4Row(wrow, qw, row)
		for b := 0; b < batch; b++ {
			out[b*qw.OutDim+row] = simd.Sdot(x[b*qw.InDim:(b+1)*qw.InDim], wrow)
		}
	}
	return true
}
