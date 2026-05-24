package gpu

import "math"

// DevAttentionFull dispatches non-causal multi-head attention for encoder use.
// This is the DevBuf-style entry point that mirrors the existing causal attention
// but without the causal mask.
//
// q, k, v: flat [seqLen * dModel]
// out: flat [seqQ * dModel]
//
// TODO: When PTX kernel is implemented, this will try GPU first, then fall back to CPU.
func DevAttentionFull(out, q, k, v []float32, seqQ, seqKV, numHeads, headDim int, scale float32) {
	dModel := numHeads * headDim
	if len(q) < seqQ*dModel || len(k) < seqKV*dModel || len(v) < seqKV*dModel || len(out) < seqQ*dModel {
		return
	}
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}

	// CPU implementation (GPU PTX pending)
	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		for tq := 0; tq < seqQ; tq++ {
			qOff := tq*dModel + hOff
			scores := make([]float32, seqKV)

			for tkv := 0; tkv < seqKV; tkv++ {
				kOff := tkv*dModel + hOff
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[qOff+d] * k[kOff+d]
				}
				scores[tkv] = dot * scale
			}

			softmaxSlice(scores)

			oOff := tq*dModel + hOff
			for tkv := 0; tkv < seqKV; tkv++ {
				vOff := tkv*dModel + hOff
				w := scores[tkv]
				for d := 0; d < headDim; d++ {
					out[oOff+d] += w * v[vOff+d]
				}
			}
		}
	}
}
