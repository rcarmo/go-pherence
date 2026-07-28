//go:build !amd64

package simd

func bf16DotF32x4(w []uint16, x []float32, cols int) (float32, float32, float32, float32) {
	return bf16DotF32x4Scalar(w, x, cols)
}

func bf16DotBF16x4(w []uint16, x []uint16, cols int) (float32, float32, float32, float32) {
	return bf16DotBF16x4Scalar(w, x, cols)
}
