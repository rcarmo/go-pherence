//go:build !riscv64

package ideogram4

func k3LayerNormNoAffine(_, _ []float32, _ float32) bool { return false }
