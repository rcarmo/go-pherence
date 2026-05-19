package model

import (
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func gqaAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	if headDim <= 0 {
		return nil
	}
	return gqaAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))))
}

func gqaAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	return simd.GQAAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
}

func gqaAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	simd.GQAAttentionScaleInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
}
