//go:build !riscv64

package diffusiongemma

func k3SelfConditioningSoftEmbeddingQ80(_ []float32, _ [][]float32, _ *TextWeights, _ *TensorBinding, _, _, _ int, _ float32) (bool, error) {
	return false, nil
}

func k3PreloadQ80TransposedBinding(_ *TextWeights, _ *TensorBinding) (bool, error) { return false, nil }
