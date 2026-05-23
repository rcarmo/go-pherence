//go:build !riscv64

package nvfp4

func dequantNVFP4Accelerated(out []float32, qw *NVFP4Weight) {
	dequantNVFP4Scalar(out, qw)
}
