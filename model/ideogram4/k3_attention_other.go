//go:build !riscv64

package ideogram4

func k3FullAttention(_, _, _, _ []float32, _, _, _ int, _ float32) bool { return false }
func k3QwenGQA(_, _, _, _ []float32, _, _, _, _ int, _ float32) bool    { return false }
