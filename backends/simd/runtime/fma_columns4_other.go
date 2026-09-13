//go:build !amd64

package simd

func fmaColumns4F32(dst, x, weight []float32) bool {
	fmaColumns4Scalar(dst, x, weight)
	return true
}
