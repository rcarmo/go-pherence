package qwen

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func Qwen35RotaryHalf(meta loaderconfig.QwenNativeMTPMetadata) int {
	headDim := meta.HeadDim
	if headDim <= 0 {
		return 0
	}
	factor := meta.PartialRotaryFactor
	if factor <= 0 || factor > 1 {
		factor = 1
	}
	rotDims := int(float64(headDim) * factor)
	if rotDims < 2 {
		rotDims = headDim
	}
	if rotDims%2 != 0 {
		rotDims--
	}
	return rotDims / 2
}

func Qwen35UseMRoPE(meta loaderconfig.QwenNativeMTPMetadata) bool {
	return meta.MRoPEInterleaved && len(meta.MRoPESection) > 0
}

func NewQwen35RoPEFreqs(meta loaderconfig.QwenNativeMTPMetadata, maxSeq int) []float32 {
	rotHalf := Qwen35RotaryHalf(meta)
	return simd.BuildRoPEFreqs(maxSeq, rotHalf, meta.HeadDim, meta.RopeTheta)
}
