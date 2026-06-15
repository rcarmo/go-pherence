//go:build !amd64

package simd

const HasDotI8F32SIMD = false

func dotI8F32(q []byte, x []float32) float32 { return dotI8F32Scalar(q, x) }

func dotI8F32x4(q []byte, x []float32, stride int) (float32, float32, float32, float32) {
	return dotI8F32x4Scalar(q, x, stride)
}
