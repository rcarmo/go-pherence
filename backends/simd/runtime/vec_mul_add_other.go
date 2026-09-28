//go:build !amd64 && !arm64

package simd

func vecMulAdd(dst, a, b []float32) { vecMulAddScalar(dst, a, b) }
