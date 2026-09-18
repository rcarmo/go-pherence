//go:build !amd64

package simd

func geluErfF32To(dst, src []float32) {
	geluErfF32ToScalar(dst, src)
}
