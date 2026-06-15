//go:build amd64

package simd

const HasDotI8F32SIMD = true

//go:noescape
func dotI8F32Asm(q []byte, x []float32) float32

//go:noescape
func dotI8F32x4Asm(q []byte, x []float32, stride int) (float32, float32, float32, float32)

func dotI8F32(q []byte, x []float32) float32 {
	if len(q)%8 != 0 {
		return dotI8F32Scalar(q, x)
	}
	return dotI8F32Asm(q, x)
}

func dotI8F32x4(q []byte, x []float32, stride int) (float32, float32, float32, float32) {
	if len(q)%8 != 0 {
		return dotI8F32x4Scalar(q, x, stride)
	}
	return dotI8F32x4Asm(q, x, stride)
}
