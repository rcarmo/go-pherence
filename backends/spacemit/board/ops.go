package board

// OpBackend is the common interface all three tiers implement.
// Ops map to transformer building blocks used in each forward pass layer.
// All slices are float32 row-major; callers own the memory.
type OpBackend interface {
	// Name returns the backend tier name.
	Name() string

	// GemvF32 computes out = W · x, where W is [outDim × inDim] row-major.
	GemvF32(out, x, w []float32, inDim, outDim int) error

	// RMSNormF32 applies RMSNorm in-place on x with weight w (len(x)==len(w)).
	RMSNormF32(x, w []float32, eps float32) error

	// RMSNormNoScaleF32 applies RMSNorm in-place without a learnable weight.
	RMSNormNoScaleF32(x []float32, eps float32) error

	// SiLUMulF32 computes dst[i] = silu(gate[i]) * up[i].
	SiLUMulF32(dst, gate, up []float32) error

	// GELUTanhMulF32 computes dst[i] = gelu_tanh(gate[i]) * up[i].
	// dst may alias gate.
	GELUTanhMulF32(dst, gate, up []float32) error

	// RoPEPartialF32 applies rotary embeddings to x in-place.
	// x is [nHeads × headDim]; freqs is [rotHalf].
	RoPEPartialF32(x, freqs []float32, pos, nHeads, headDim, rotHalf int) error

	// AttentionScoresF32 computes scaled QK^T attention logits.
	// q is [nHeads × headDim], kCache is [seqLen × nKVHeads × headDim],
	// out is [nHeads × seqLen] (raw logits, no softmax applied).
	AttentionScoresF32(out, q, kCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error
}
