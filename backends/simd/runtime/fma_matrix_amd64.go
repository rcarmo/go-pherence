//go:build amd64

package simd

//go:noescape
func fmaMatrixAsm(dst, a, b []float32, m, n, k int) bool

func fmaMatrixF32(dst, a, b []float32, m, n, k int) bool {
	if !HasAffineF32Asm() {
		fmaMatrixScalar(dst, a, b, m, n, k)
		return true
	}
	// Guard, zeroing and dispatch share one assembly call/FP environment.
	return fmaMatrixAsm(dst, a, b, m, n, k)
}
