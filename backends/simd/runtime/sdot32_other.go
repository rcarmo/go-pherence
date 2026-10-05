//go:build !amd64

package simd

func sdot32(x, y []float32) float32 { return sdot32Scalar(x, y) }
