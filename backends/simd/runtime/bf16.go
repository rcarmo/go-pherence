package simd

import (
	"unsafe"

	bf16kernel "github.com/rcarmo/go-pherence/backends/simd/runtime/bf16"
)

type BF16 = bf16kernel.BF16

func BF16ToF32(b BF16) float32                   { return bf16kernel.BF16ToF32(b) }
func F32ToBF16(f float32) BF16                   { return bf16kernel.F32ToBF16(f) }
func BF16DotF32(x []uint16, y []float32) float32 { return bf16kernel.BF16DotF32(x, y) }
func BF16DotF32Checked(x []uint16, y []float32) (float32, bool) {
	return bf16kernel.BF16DotF32Checked(x, y)
}
func BF16Dot(x, y []uint16) float32 { return bf16kernel.BF16Dot(x, y) }
func BF16DotChecked(x, y []uint16) (float32, bool) {
	return bf16kernel.BF16DotChecked(x, y)
}
func BF16RMSNorm(x, w []uint16, eps float32) { bf16kernel.BF16RMSNorm(x, w, eps) }
func BF16RMSNormChecked(x, w []uint16, eps float32) bool {
	return bf16kernel.BF16RMSNormChecked(x, w, eps)
}
func BF16VecAdd(dst, a, b []uint16) { bf16kernel.BF16VecAdd(dst, a, b) }
func BF16VecAddChecked(dst, a, b []uint16) bool {
	return bf16kernel.BF16VecAddChecked(dst, a, b)
}
func BF16GemvNT(out []uint16, x []uint16, w []float32, inDim, outDim int) {
	bf16kernel.BF16GemvNT(out, x, w, inDim, outDim)
}
func BF16GemvNTChecked(out []uint16, x []uint16, w []float32, inDim, outDim int) bool {
	return bf16kernel.BF16GemvNTChecked(out, x, w, inDim, outDim)
}
func BF16FromF32Slice(f32 []float32) []uint16 { return bf16kernel.BF16FromF32Slice(f32) }
func BF16ToF32Slice(bf16 []uint16) []float32  { return bf16kernel.BF16ToF32Slice(bf16) }
func BF16SlicePtr(s []uint16) unsafe.Pointer  { return bf16kernel.BF16SlicePtr(s) }
