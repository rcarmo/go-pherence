//go:build !riscv64

package ideogram4

func k3SiLUTo(_, _ []float32) bool                                 { return false }
func k3MulTo(_, _, _ []float32) bool                               { return false }
func k3SiLUMulInPlace(_, _ []float32) bool                         { return false }
func k3MLPBatch(DiTLayer, []float32, []float32, int) (bool, error) { return false, nil }
