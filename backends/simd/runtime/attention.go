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

// GQAAttentionScaleInto computes grouped-query attention into caller-owned
// buffers. It preserves the historical no-op-on-malformed-input behavior.
func GQAAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	kernels.GQAAttentionScaleInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, Sdot, Saxpy)
}

// GQAAttentionScaleTo computes grouped-query attention into caller-owned
// buffers and reports malformed inputs.
func GQAAttentionScaleTo(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) bool {
	if seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 {
		return false
	}
	h, okH := checkedMulInt(numHeads, headDim)
	kvDim, okKV := checkedMulInt(numKVHeads, headDim)
	kvTotal, okTotal := checkedMulInt(seqLen, kvDim)
	if !okH || !okKV || !okTotal || len(out) < h || len(scores) < seqLen || len(q) < h || len(kCache) < kvTotal || len(vCache) < kvTotal {
		return false
	}
	kernels.GQAAttentionScaleInto(out[:h], scores[:seqLen], q[:h], kCache[:kvTotal], vCache[:kvTotal], seqLen, numHeads, numKVHeads, headDim, scale, Sdot, Saxpy)
	return true
}
