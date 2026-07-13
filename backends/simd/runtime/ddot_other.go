//go:build !amd64

package simd

// Ddot computes the float64 dot product over the common input prefix.
func Ddot(x, y []float64) float64 { return ddotGo(x, y) }
