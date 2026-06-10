//go:build !riscv64

package ideogram4

func k3MRoPEToQK(_, _ []float32, _ *MRoPE, _, _, _ int) bool { return false }
func k3QwenRoPE(_ []float32, _ ropeTable, _ int) bool        { return false }
