//go:build amd64

package simd

//go:noescape
func fmaColumnsAsm(dst, x, weight []float32) bool

func fmaColumnsF32(dst, x, weight []float32) bool {
	if HasAffineF32Asm() {
		return fmaColumnsAsm(dst, x, weight)
	}
	fmaColumnsScalar(dst, x, weight)
	return true
}
