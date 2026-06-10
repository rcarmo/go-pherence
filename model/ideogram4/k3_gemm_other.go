//go:build !riscv64

package ideogram4

func k3GemmRowsF32(_ []float32, _, _ []float32, _, _, _ int) bool { return false }
