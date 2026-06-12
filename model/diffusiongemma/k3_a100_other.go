//go:build !riscv64

package diffusiongemma

func k3A100Q8Enabled() bool { return false }
func k3Threads() int        { return 1 }

func k3GemmRowsQ80(_ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3Gemm2RowsQ80(_ []float32, _ []float32, _ []float32, _ int, _ *TextWeights, _ *TensorBinding, _ *TensorBinding) (bool, error) {
	return false, nil
}

func k3GemmManyRowsQ80(_ [][]float32, _ []float32, _ int, _ *TextWeights, _ []*TensorBinding) (bool, error) {
	return false, nil
}
