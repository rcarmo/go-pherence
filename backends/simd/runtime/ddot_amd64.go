//go:build amd64

package simd

//go:noescape
func ddotAsm(x, y []float64) float64

// Ddot computes the float64 dot product over the common input prefix.
func Ddot(x, y []float64) float64 {
	if len(x) > 0 && len(x) == len(y) && HasVecAsm {
		return ddotAsm(x, y)
	}
	return ddotGo(x, y)
}
