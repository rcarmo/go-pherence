//go:build !riscv64

package nvfp4

func gemmNVFP4Accelerated(out, x []float32, batch int, qw *NVFP4Weight) bool {
	for b := 0; b < batch; b++ {
		gemvNVFP4Rows(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw, 0, qw.OutDim)
	}
	return true
}
