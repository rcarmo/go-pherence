//go:build !riscv64

package ideogram4

type k3FP8Cache struct{}

func (c *k3FP8Cache) release() {}

func k3FP8Batch(_ *FP8Linear, _, _ []float32, _ int) (bool, error) { return false, nil }
