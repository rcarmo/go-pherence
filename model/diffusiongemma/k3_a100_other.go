//go:build !riscv64

package diffusiongemma

func k3A100Q8Enabled() bool     { return false }
func k3A100LMHeadEnabled() bool { return false }
func k3Threads() int            { return 1 }

func k3EvictQ80Tensor(_ *TextWeights, _ string) bool { return false }
func k3EvictQ80Layer(_ *TextWeights, _ int) int      { return 0 }
func k3ClearQ80CacheForWeights(_ *TextWeights)       {}
func k3Q80CacheStats(_ *TextWeights) (int, int64)    { return 0, 0 }

func k3GemmRowsQ80(_ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3Gemm2RowsQ80(_ []float32, _ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3GemmManyRowsQ80(_ [][]float32, _ []float32, _ int, _ *TextWeights, _ []*TensorBinding) (bool, error) {
	return false, nil
}

func (w *TextWeights) PreloadLayerQ80(_ int, _ bool) (int, error) { return 0, nil }
func (w *TextWeights) PreloadLayerRangeQ80(_ int, _ int, _ bool) (int, error) {
	return 0, nil
}
