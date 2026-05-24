package gpu

import "math"

// FullAttention computes non-causal multi-head attention (for encoder).
// q, k, v: [seqLen * dModel] flat
// Returns [seqLen * dModel].
//
// TODO: GPU PTX fast path. Currently CPU-only.
func FullAttention(out, q, k, v []float32, seqQ, seqKV, numHeads, headDim int) {
	dModel := numHeads * headDim
	if len(q) < seqQ*dModel || len(k) < seqKV*dModel || len(v) < seqKV*dModel || len(out) < seqQ*dModel {
		return
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

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

// CrossAttention computes cross-attention: Q from decoder, K/V from encoder.
// q: [decLen * dModel], k/v: [encLen * dModel]
// Returns [decLen * dModel].
//
// TODO: GPU PTX fast path. Currently CPU-only.
func CrossAttention(out, q, k, v []float32, decLen, encLen, numHeads, headDim int) {
	FullAttention(out, q, k, v, decLen, encLen, numHeads, headDim)
}

func softmaxSlice(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range x {
			x[i] /= sum
		}
	}
}
