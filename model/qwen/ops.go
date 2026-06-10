package qwen

import "github.com/rcarmo/go-pherence/backends/simd/runtime"

func rmsNormInPlace(x, weight []float32, eps float32) {
	simd.RMSNorm(x, weight, eps)
}
