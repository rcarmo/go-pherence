//go:build !amd64

package simd

func expF32To(dst, src []float32) {
	expF32ToScalar(dst, src)
}
