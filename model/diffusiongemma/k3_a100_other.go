//go:build !riscv64

package diffusiongemma

func k3A100Q8Enabled() bool             { return false }
func k3A100LMHeadEnabled() bool         { return false }
func k3A100LMHeadPrefetchEnabled() bool { return false }
func k3A100LMHeadCandidates(topK, vocab int) int {
	if topK > vocab {
		return vocab
	}
	return topK
}
func k3Q80PrefetchEnabled() bool         { return false }
func k3Q80PrefetchExperts() bool         { return false }
func k3Q80SelectedPrefetchEnabled() bool { return false }
func k3Threads() int                     { return 1 }

func k3EvictQ80Tensor(_ *TextWeights, _ string) bool { return false }
func k3EvictQ80Layer(_ *TextWeights, _ int) int      { return 0 }
func k3ClearQ80CacheForWeights(_ *TextWeights)       {}
func k3Q80CacheStats(_ *TextWeights) (int, int64)    { return 0, 0 }

func k3PreloadQ80Binding(_ *TextWeights, _ *TensorBinding) (bool, error) { return false, nil }

func k3GemmRowsQ80(_ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3Gemm2RowsQ80(_ []float32, _ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3GemmManyRowsQ80(_ [][]float32, _ []float32, _ int, _ *TextWeights, _ []*TensorBinding) (bool, error) {
	return false, nil
}

func EstimateQ80ResidencyBudgetFromWeights(weights *TextWeights, _ bool, budgetBytes int64) ResidencyBudget {
	out := ResidencyBudget{BudgetBytes: budgetBytes}
	if weights != nil {
		out.TotalLayers = len(weights.Layers)
	}
	return out
}

func (w *TextWeights) PreloadLayerQ80(_ int, _ bool) (int, error) { return 0, nil }
func (w *TextWeights) PreloadLayerRangeQ80(_ int, _ int, _ bool) (int, error) {
	return 0, nil
}
