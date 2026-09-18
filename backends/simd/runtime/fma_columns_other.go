//go:build !amd64

package simd

func fmaColumnsF32(dst, x, weight []float32) bool {
	fmaColumnsScalar(dst, x, weight)
	return true
}
