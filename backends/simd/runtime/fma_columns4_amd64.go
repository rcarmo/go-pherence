//go:build amd64

package simd

//go:noescape
func fmaColumns4Asm(dst, x, weight []float32) bool

func fmaColumns4F32(dst, x, weight []float32) bool {
	if HasAffineF32Asm() {
		return fmaColumns4Asm(dst, x, weight)
	}
	fmaColumns4Scalar(dst, x, weight)
	return true
}
