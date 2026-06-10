package kernels

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
)

func GQAAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, dot func([]float32, []float32) float32, saxpy func(float32, []float32, []float32)) []float32 {
	if headDim <= 0 {
		return nil
	}
	return GQAAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))), dot, saxpy)
}

func GQAAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32, dot func([]float32, []float32) float32, saxpy func(float32, []float32, []float32)) []float32 {
	if numHeads <= 0 || headDim <= 0 {
		return nil
	}
	out := make([]float32, numHeads*headDim)
	if seqLen <= 0 {
		return out
	}
	scores := make([]float32, seqLen)
	GQAAttentionScaleInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, dot, saxpy)
	return out
}

func GQAAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32, dot func([]float32, []float32) float32, saxpy func(float32, []float32, []float32)) {
	if seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 || dot == nil || saxpy == nil {
		return
	}
	h, okH := checked.MulInt(numHeads, headDim)
	kvDim, okKV := checked.MulInt(numKVHeads, headDim)
	kvTotal, okTotal := checked.MulInt(seqLen, kvDim)
	if !okH || !okKV || !okTotal || len(out) < h || len(scores) < seqLen || len(q) < h || len(kCache) < kvTotal || len(vCache) < kvTotal {
		return
	}
	headsPerKV := numHeads / numKVHeads
	out = out[:h]
	clear(out)
	scores = scores[:seqLen]

	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV

		qHead := q[head*headDim : (head+1)*headDim]
		for t := 0; t < seqLen; t++ {
			kHead := kCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			scores[t] = dot(qHead, kHead) * scale
		}

		mx := scores[0]
		for _, v := range scores[1:] {
			if v > mx {
				mx = v
			}
		}
		expSum := float32(0)
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - mx)))
			expSum += scores[i]
		}
		inv := 1.0 / expSum
		for i := range scores {
			scores[i] *= inv
		}

		outHead := out[head*headDim : (head+1)*headDim]
		for t := 0; t < seqLen; t++ {
			vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			saxpy(scores[t], vHead, outHead)
		}
	}
}
