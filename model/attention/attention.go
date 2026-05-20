package attention

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
	out, ok := simd.GQAAttentionScaleChecked(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
	if !ok {
		return nil
	}
	return out
}

func gqaAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	_ = simd.GQAAttentionScaleTo(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
}
