//go:build !riscv64

package nvfp4

func gemvNVFP4Accelerated(out, x []float32, qw *NVFP4Weight) {
	gemvNVFP4Rows(out, x, qw, 0, qw.OutDim)
}
