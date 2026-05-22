//go:build riscv64

package nvfp4

func dequantNVFP4Accelerated(out []float32, qw *NVFP4Weight) {
	for row := 0; row < qw.OutDim; row++ {
		dequantNVFP4WeightRow(out[row*qw.InDim:(row+1)*qw.InDim], qw, row)
	}
}
