//go:build !riscv64

package diffusiongemma

func k3RunPerExpertA100(_ *TextWeights, _ TextLayerBindings, _ expertWeightLayout, _ ForwardScratch, _ []float32, _, _, _ int) (bool, error) {
	return false, nil
}

func k3RunPerExpertRowsA100(_ *TextWeights, _ expertWeightLayout, _ []float32, _ []int, _ []float32, _ []float32, _ []float32, _, _, _ int) (bool, error) {
	return false, nil
}
