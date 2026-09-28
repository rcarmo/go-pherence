//go:build arm64

package simd

//go:noescape
func fmaMatrixArm64Asm(dst, a, b []float32, m, n, k int)

// fmaMatrixF32 uses serial-K NEON FMA and overwrites dst in a single pass.
// The checked caller validates extents, finiteness and non-overlap first.
func fmaMatrixF32(dst, a, b []float32, m, n, k int) bool {
	if !HasSgemmAsm {
		fmaMatrixScalar(dst, a, b, m, n, k)
		return true
	}
	fmaMatrixArm64Asm(dst, a, b, m, n, k)
	return true
}
