//go:build !amd64

package simd

const HasDotRowsx4SIMD = false

func dotRowsx4(w []float32, x []float32, cols int) (float32, float32, float32, float32) {
	return dotRowsx4Scalar(w, x, cols)
}
