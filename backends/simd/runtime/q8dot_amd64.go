//go:build amd64

package simd

const HasDotI8F32SIMD = true

//go:noescape
func dotI8F32Asm(q []byte, x []float32) float32

func dotI8F32(q []byte, x []float32) float32 {
	if len(q)%8 != 0 {
		return dotI8F32Scalar(q, x)
	}
	return dotI8F32Asm(q, x)
}
