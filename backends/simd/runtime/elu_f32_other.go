//go:build !amd64

package simd

func eluF32To(dst, src []float32) {
	eluF32ToScalar(dst, src)
}
