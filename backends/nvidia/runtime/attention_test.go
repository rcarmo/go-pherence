package nvidia

import (
	"math"
	"testing"
)

func TestDevAttentionMHA(t *testing.T) {
	if !SgemmReady() {
		t.Skip("GPU not available")
	}
	seqLen, nHeads, headDim := 5, 3, 4
	nKVHeads := nHeads
	dModel := nHeads * headDim
	q := make([]float32, dModel)
	k := make([]float32, seqLen*dModel)
	v := make([]float32, seqLen*dModel)
	for i := range q {
		q[i] = float32(i%7-3) * 0.17
	}
	for i := range k {
		k[i] = float32(i%11-5) * 0.09
		v[i] = float32(i%13-6) * 0.07
	}
	want := attentionCPU(q, k, v, seqLen, nHeads, nKVHeads, headDim, 1/float32(math.Sqrt(float64(headDim))))
	out := NewDevBuf(dModel)
	qBuf := NewDevBufFrom(q)
	kBuf := NewDevBufFrom(k)
	vBuf := NewDevBufFrom(v)
	DevAttention(out, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, 1/float32(math.Sqrt(float64(headDim))))
	got := out.Data()[:dModel]
	for i := range want {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 2e-3 {
			t.Fatalf("idx %d got %.6f want %.6f diff %.6f", i, got[i], want[i], diff)
		}
	}
}

func attentionCPU(q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, nHeads*headDim)
	headsPerKV := nHeads / nKVHeads
	kvDim := nKVHeads * headDim
	for h := 0; h < nHeads; h++ {
		kvh := h / headsPerKV
		scores := make([]float32, seqLen)
		maxScore := float32(math.Inf(-1))
		for s := 0; s < seqLen; s++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[h*headDim+d] * k[s*kvDim+kvh*headDim+d]
			}
			scores[s] = dot * scale
			if scores[s] > maxScore {
				maxScore = scores[s]
			}
		}
		var sum float32
		for s := range scores {
			scores[s] = float32(math.Exp(float64(scores[s] - maxScore)))
			sum += scores[s]
		}
		for s := range scores {
			w := scores[s] / sum
			for d := 0; d < headDim; d++ {
				out[h*headDim+d] += w * v[s*kvDim+kvh*headDim+d]
			}
		}
	}
	return out
}
