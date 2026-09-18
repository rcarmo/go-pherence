//go:build !amd64

package simd

func sinF32To(dst, src []float32) {
	sinF32ToScalar(dst, src)
}
