package gpu

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
)

// DevAttentionFull dispatches non-causal multi-head attention for encoder use.
// This is the DevBuf-style entry point that mirrors the existing causal attention
// but without the causal mask.
//
// q, k, v: flat [seqLen * dModel]
// out: flat [seqQ * dModel]
//
// Currently CPU SIMD only. Output is overwritten and must not overlap inputs.
// Malformed dimensions/buffers return without writing output.
func DevAttentionFull(out, q, k, v []float32, seqQ, seqKV, numHeads, headDim int, scale float32) {
	dModel, ok := checked.MulInt(numHeads, headDim)
	qLen, okQ := checked.MulInt(seqQ, dModel)
	kvLen, okKV := checked.MulInt(seqKV, dModel)
	if seqQ <= 0 || seqKV <= 0 || numHeads <= 0 || headDim <= 0 || !ok || !okQ || !okKV || len(q) < qLen || len(k) < kvLen || len(v) < kvLen || len(out) < qLen || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return
	}
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}

	// One reusable score row per call, rather than one per head/query. The
	// checked SIMD kernel preserves shared GQA shape and softmax semantics.
	scores := make([]float32, seqKV)
	for tq := 0; tq < seqQ; tq++ {
		start := tq * dModel
		_ = simd.GQAAttentionScaleTo(out[start:start+dModel], scores, q[start:start+dModel], k[:kvLen], v[:kvLen], seqKV, numHeads, numHeads, headDim, scale)
	}
}
