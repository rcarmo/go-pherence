//go:build amd64

package simd

//go:noescape
func sdot32Asm(x, y []float32) float32

func sdot32(x, y []float32) float32 {
	if HasDotAsm {
		return sdot32Asm(x, y)
	}
	return sdot32Scalar(x, y)
}
