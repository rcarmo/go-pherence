package k3

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SIMDBackend routes ops through the RVV-tuned CPU SIMD paths.
// It is the always-available fallback and never returns an error.
type SIMDBackend struct{}

func (SIMDBackend) Name() string { return TierCPU.String() }

// GemvF32: W is [outDim × inDim], GemvRows(out, x, w, rows=outDim, cols=inDim).
func (SIMDBackend) GemvF32(out, x, w []float32, inDim, outDim int) error {
	simd.GemvRows(out, x, w, outDim, inDim)
	return nil
}

func (SIMDBackend) RMSNormF32(x, w []float32, eps float32) error {
	simd.RMSNorm(x, w, eps)
	return nil
}

func (SIMDBackend) RMSNormNoScaleF32(x []float32, eps float32) error {
	simd.RMSNormNoScale(x, eps)
	return nil
}

func (SIMDBackend) SiLUMulF32(dst, gate, up []float32) error {
	simd.SiLUMul(dst, gate, up)
	return nil
}

// GELUTanhMulF32: uses GELUTanhMulTo(dst, gate, up) → dst[i] = gelu_tanh(gate[i])*up[i].
func (SIMDBackend) GELUTanhMulF32(dst, gate, up []float32) error {
	simd.GELUTanhMulTo(dst, gate, up)
	return nil
}

func (SIMDBackend) RoPEPartialF32(x, freqs []float32, pos, nHeads, headDim, rotHalf int) error {
	simd.ApplyRoPEPartial(x, freqs, pos, nHeads, headDim, rotHalf)
	return nil
}

// AttentionScoresF32 computes scaled QK^T logits via a simple dot-product loop.
// This avoids pulling in vCache (not available at this interface level).
func (SIMDBackend) AttentionScoresF32(out, q, kCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	if nKVHeads == 0 || nHeads == 0 {
		return nil
	}
	groupSize := nHeads / nKVHeads
	if groupSize == 0 {
		groupSize = 1
	}
	for h := 0; h < nHeads; h++ {
		kvHead := h / groupSize
		qRow := q[h*headDim : (h+1)*headDim]
		for t := 0; t < seqLen; t++ {
			kRow := kCache[(t*nKVHeads+kvHead)*headDim : (t*nKVHeads+kvHead+1)*headDim]
			var sum float32
			for d := 0; d < headDim; d++ {
				sum += qRow[d] * kRow[d]
			}
			out[h*seqLen+t] = sum * scale
		}
	}
	return nil
}
