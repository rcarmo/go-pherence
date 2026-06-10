//go:build !riscv64

package ideogram4

func k3RMSNormWeighted(_ []float32, _, _ []float32, _ float32) bool { return false }
