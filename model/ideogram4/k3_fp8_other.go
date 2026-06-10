//go:build !riscv64

package ideogram4

func k3FP8Batch(_ *FP8Linear, _, _ []float32, _ int) (bool, error) { return false, nil }
