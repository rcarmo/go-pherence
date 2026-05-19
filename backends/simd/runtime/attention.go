package simd

import (
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/kernels"
)

func GQAAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	if headDim <= 0 {
		return nil
	}
	return GQAAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))))
}

func GQAAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	return kernels.GQAAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, Sdot, Saxpy)
}

func GQAAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	kernels.GQAAttentionScaleInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, Sdot, Saxpy)
}
