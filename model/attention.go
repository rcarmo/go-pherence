package model

import (
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func attentionLogitSoftcap(cfg LlamaConfig) float32 {
	if cfg.ModelType != "gemma2" && cfg.ModelType != "gemma2_text" {
		return 0
	}
	return float32(cfg.AttentionLogitSoftcapping)
}

func gqaAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	if headDim <= 0 {
		return nil
	}
	return gqaAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))))
}

func gqaAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	return gqaAttentionScaleSoftcap(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionScaleSoftcap(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) []float32 {
	if softcap <= 0 {
		out, ok := simd.GQAAttentionScaleChecked(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
		if !ok {
			return nil
		}
		return out
	}
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	if !gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, softcap) {
		return nil
	}
	return out
}

func gqaAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	_ = gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) bool {
	if softcap <= 0 {
		return simd.GQAAttentionScaleTo(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
	}
	if seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 || len(out) < numHeads*headDim || len(scores) < seqLen || len(q) < numHeads*headDim || len(kCache) < seqLen*numKVHeads*headDim || len(vCache) < seqLen*numKVHeads*headDim {
		return false
	}
	groupSize := numHeads / numKVHeads
	kvDim := numKVHeads * headDim
	invCap := float32(1) / softcap
	for head := 0; head < numHeads; head++ {
		kvHead := head / groupSize
		qRow := q[head*headDim : (head+1)*headDim]
		for token := 0; token < seqLen; token++ {
			kOff := token*kvDim + kvHead*headDim
			score := simd.Sdot(qRow, kCache[kOff:kOff+headDim]) * scale
			scores[token] = float32(math.Tanh(float64(score*invCap))) * softcap
		}
		if !simd.SoftmaxInPlace(scores[:seqLen]) {
			return false
		}
		outRow := out[head*headDim : (head+1)*headDim]
		clear(outRow)
		for token, weight := range scores[:seqLen] {
			vOff := token*kvDim + kvHead*headDim
			vRow := vCache[vOff : vOff+headDim]
			for dim := range outRow {
				outRow[dim] += weight * vRow[dim]
			}
		}
	}
	return true
}

// gqaAttentionHeadsParallel is the heads-parallel variant used by the
// sequential autoregressive decode step. It reuses the provided scores buffer
// for the serial fallback and is bit-identical to gqaAttentionScaleInto.
func gqaAttentionHeadsParallel(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	gqaAttentionHeadsParallelSoftcap(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionHeadsParallelSoftcap(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) {
	if softcap > 0 {
		_ = gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, softcap)
		return
	}
	_ = simd.GQAAttentionHeadsParallelTo(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
}
