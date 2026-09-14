//go:build !amd64

package simd

func fmaMatrixF32(dst, a, b []float32, m, n, k int) bool {
	fmaMatrixScalar(dst, a, b, m, n, k)
	return true
}
