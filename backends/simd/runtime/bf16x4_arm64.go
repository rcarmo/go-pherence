//go:build arm64

package simd

//go:noescape
func bf16DotF32x4Arm64Asm(w []uint16, x []float32, cols int) (float32, float32, float32, float32)

func bf16DotF32x4(w []uint16, x []float32, cols int) (float32, float32, float32, float32) {
	if !HasVecAsm {
		return bf16DotF32x4Scalar(w, x, cols)
	}
	return bf16DotF32x4Arm64Asm(w, x, cols)
}

func bf16DotBF16x4(w []uint16, x []uint16, cols int) (float32, float32, float32, float32) {
	return bf16DotBF16x4Scalar(w, x, cols)
}
