//go:build !amd64

package simd

const HasSdotx4SIMD = false

func sdotx4(w, x []float32, stride int) (float32, float32, float32, float32) {
	return sdotx4Scalar(w, x, stride)
}
