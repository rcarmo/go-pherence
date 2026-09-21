package gpu

// FullAttention computes non-causal multi-head attention (for encoder).
// q, k, v: [seqLen * dModel] flat
// Returns [seqLen * dModel].
//
// TODO: GPU PTX fast path. Currently CPU-only.
func FullAttention(out, q, k, v []float32, seqQ, seqKV, numHeads, headDim int) {
	DevAttentionFull(out, q, k, v, seqQ, seqKV, numHeads, headDim, 0)
}

// CrossAttention computes cross-attention: Q from decoder, K/V from encoder.
// q: [decLen * dModel], k/v: [encLen * dModel]
// Returns [decLen * dModel].
//
// TODO: GPU PTX fast path. Currently CPU-only.
func CrossAttention(out, q, k, v []float32, decLen, encLen, numHeads, headDim int) {
	FullAttention(out, q, k, v, decLen, encLen, numHeads, headDim)
}
