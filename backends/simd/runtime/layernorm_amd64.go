//go:build amd64

package simd

//go:noescape
func layerNormAffineRowAsm(out, x, gamma, beta []float32, eps float32)

func layerNormAffineRowTo(out, x, gamma, beta []float32, eps float32) {
	if HasVecAsm && len(x) >= 8 {
		layerNormAffineRowAsm(out, x, gamma, beta, eps)
		return
	}
	layerNormAffineRowGo(out, x, gamma, beta, eps)
}
