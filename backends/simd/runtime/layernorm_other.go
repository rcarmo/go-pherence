//go:build !amd64

package simd

func layerNormAffineRowTo(out, x, gamma, beta []float32, eps float32) {
	layerNormAffineRowGo(out, x, gamma, beta, eps)
}
