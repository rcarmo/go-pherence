package whisper

import (
	"math"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Decoder implements the Whisper token decoder:
// Token embedding + positional embedding → N decoder layers (self-attn + cross-attn + MLP).
type Decoder struct {
	cfg Config

	// Token embeddings
	TokenEmbed []float32 // [vocabSize, dModel]
	PosEmbed   []float32 // [maxDecoderLength, dModel]
	lmHeadGPU  *nv.DevBuf

	// Decoder layers
	Layers []DecoderLayer

	// Final LayerNorm
	FinalLNWeight []float32
	FinalLNBias   []float32

	// Optional generation-config logit suppression.
	SuppressTokens      []int
	BeginSuppressTokens []int
}

// DecoderLayer holds weights for one Whisper decoder transformer layer.
type DecoderLayer struct {
	// Causal self-attention
	SelfAttnLNWeight []float32
	SelfAttnLNBias   []float32
	SelfQWeight      []float32
	SelfQBias        []float32
	SelfKWeight      []float32
	SelfKBias        []float32
	SelfVWeight      []float32
	SelfVBias        []float32
	SelfOWeight      []float32
	SelfOBias        []float32

	// Cross-attention (Q from decoder, K/V from encoder)
	CrossAttnLNWeight []float32
	CrossAttnLNBias   []float32
	CrossQWeight      []float32
	CrossQBias        []float32
	CrossKWeight      []float32
	CrossKBias        []float32
	CrossVWeight      []float32
	CrossVBias        []float32
	CrossOWeight      []float32
	CrossOBias        []float32

	// MLP
	MLPLNWeight []float32
	MLPLNBias   []float32
	FC1Weight   []float32
	FC1Bias     []float32
	FC2Weight   []float32
	FC2Bias     []float32

	// Optional GPU-resident decoder projection weights. These are intentionally
	// used only when explicitly enabled because per-token CUDA launch overhead can
	// dominate on short chunks.
	gpuFC1Weight *nv.DevBuf
	gpuFC2Weight *nv.DevBuf
}

// DecoderState holds cached KV for incremental decoding.
type DecoderState struct {
	// Self-attention KV cache per layer: [layer][pos * dModel]
	SelfKCache [][]float32
	SelfVCache [][]float32

	// Cross-attention KV (computed once from encoder output)
	CrossK [][]float32 // [layer][encLen * dModel]
	CrossV [][]float32 // [layer][encLen * dModel]

	// Head-major copies of the cross-attention KV ([layer][head*encLen*headDim])
	// for contiguous per-head reads in the decode hot loop.
	CrossKHead [][]float32
	CrossVHead [][]float32

	// Optional GPU-resident cross-attention KV. These are populated by
	// NewDecoderStateGPU when GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1.
	CrossKGPU []*nv.DevBuf
	CrossVGPU []*nv.DevBuf

	Pos       int          // Current token position
	LastToken int          // Last token fed into ForwardToken, or -1 before prompt
	Bufs      *decoderBufs // Reusable buffers (nil = allocate per call)
}

// NewDecoder creates a Decoder with allocated layers.
func NewDecoder(cfg Config) *Decoder {
	return &Decoder{
		cfg:    cfg,
		Layers: make([]DecoderLayer, cfg.DecoderLayers),
	}
}

// NewDecoderState initializes decoding state for incremental generation.
func NewDecoderState(cfg Config, encoderOutput []float32, encLen int, dec *Decoder) *DecoderState {
	dModel := cfg.DecoderDModel
	numLayers := cfg.DecoderLayers

	state := &DecoderState{
		SelfKCache: make([][]float32, numLayers),
		SelfVCache: make([][]float32, numLayers),
		CrossK:     make([][]float32, numLayers),
		CrossV:     make([][]float32, numLayers),
		CrossKHead: make([][]float32, numLayers),
		CrossVHead: make([][]float32, numLayers),
		LastToken:  -1,
		Bufs:       newDecoderBufs(cfg),
	}

	// Pre-allocate self-attention KV caches
	for l := 0; l < numLayers; l++ {
		state.SelfKCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
		state.SelfVCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
	}

	// Pre-compute cross-attention K/V from encoder output (done once)
	// Use GPU SGEMM if available for this large batched matmul
	for l := 0; l < numLayers; l++ {
		layer := &dec.Layers[l]
		state.CrossK[l] = linearForwardOpt(encoderOutput, layer.CrossKWeight, layer.CrossKBias, encLen, dModel, dModel)
		state.CrossV[l] = linearForwardOpt(encoderOutput, layer.CrossVWeight, layer.CrossVBias, encLen, dModel, dModel)
		// Reorder once to head-major so each decoded token reads each head's
		// frames contiguously instead of stride-dModel (the decode bottleneck).
		state.CrossKHead[l] = toHeadMajor(state.CrossK[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
		state.CrossVHead[l] = toHeadMajor(state.CrossV[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
	}

	return state
}

// ForwardToken runs one decoder step for a single token.
// Returns logits [vocabSize].
var (
	decSelfNs  int64
	decCrossNs int64
	decMlpNs   int64
	decLmNs    int64
)

func (dec *Decoder) ForwardToken(tokenID int, state *DecoderState) []float32 {
	cfg := dec.cfg
	dModel := cfg.DecoderDModel
	pos := state.Pos

	bufs := state.Bufs
	if bufs == nil {
		bufs = newDecoderBufs(cfg)
		state.Bufs = bufs
	}

	// Token embedding + positional embedding
	x := bufs.x
	zeroFloat32s(x)
	if tokenID >= 0 && tokenID < cfg.VocabSize && dec.TokenEmbed != nil {
		copy(x, dec.TokenEmbed[tokenID*dModel:(tokenID+1)*dModel])
	}
	if dec.PosEmbed != nil && pos < cfg.MaxDecoderLength {
		for d := 0; d < dModel; d++ {
			x[d] += dec.PosEmbed[pos*dModel+d]
		}
	}

	numHeads := cfg.DecoderHeads
	headDim := cfg.HeadDim
	encLen := 0
	if len(state.CrossK) > 0 {
		encLen = len(state.CrossK[0]) / dModel
	}

	for l := range dec.Layers {
		layer := &dec.Layers[l]

		// --- Causal self-attention ---
		tphase := nowNs()
		layerNormInto(bufs.normed, x, layer.SelfAttnLNWeight, layer.SelfAttnLNBias, dModel)
		linearInto(bufs.q, bufs.normed, layer.SelfQWeight, layer.SelfQBias, dModel, dModel)
		linearInto(bufs.k, bufs.normed, layer.SelfKWeight, layer.SelfKBias, dModel, dModel)
		linearInto(bufs.v, bufs.normed, layer.SelfVWeight, layer.SelfVBias, dModel, dModel)

		// Append to KV cache
		state.SelfKCache[l] = append(state.SelfKCache[l], bufs.k...)
		state.SelfVCache[l] = append(state.SelfVCache[l], bufs.v...)

		// Causal attention for a single current query: with only one query row, it
		// attends to the cached prefix (0..pos), so the non-causal GPU attention
		// wrapper is equivalent. Keep CPU/SIMD as the default oracle/fallback.
		seqKV := pos + 1
		if !bufs.selfAttentionGPU(bufs.selfOut, bufs.q, state.SelfKCache[l], state.SelfVCache[l], seqKV, numHeads, headDim) {
			attentionSingleInto(bufs.selfOut, bufs.q, state.SelfKCache[l], state.SelfVCache[l], seqKV, numHeads, headDim, bufs.scores)
		}

		linearInto(bufs.proj, bufs.selfOut, layer.SelfOWeight, layer.SelfOBias, dModel, dModel)
		for d := range x {
			x[d] += bufs.proj[d]
		}
		decSelfNs += nowNs() - tphase

		// --- Cross-attention ---
		tphase = nowNs()
		layerNormInto(bufs.normed, x, layer.CrossAttnLNWeight, layer.CrossAttnLNBias, dModel)
		linearInto(bufs.crossQ, bufs.normed, layer.CrossQWeight, layer.CrossQBias, dModel, dModel)

		// Cross-attention: Q from decoder, K/V from encoder (full, non-causal)
		if !bufs.attentionGPU(bufs.crossOut, bufs.crossQ, state.CrossKGPU, state.CrossVGPU, l, encLen, numHeads, headDim) {
			crossAttentionHeadMajor(bufs.crossOut, bufs.crossQ, state.CrossKHead[l], state.CrossVHead[l], encLen, numHeads, headDim, bufs.scores)
		}
		linearInto(bufs.crossProj, bufs.crossOut, layer.CrossOWeight, layer.CrossOBias, dModel, dModel)
		for d := range x {
			x[d] += bufs.crossProj[d]
		}
		decCrossNs += nowNs() - tphase

		// --- MLP ---
		tphase = nowNs()
		layerNormInto(bufs.mlpIn, x, layer.MLPLNWeight, layer.MLPLNBias, dModel)
		if !bufs.linearGPU(bufs.hidden, bufs.mlpIn, layer.gpuFC1Weight, layer.FC1Bias, dModel, cfg.DecoderFFNDim) {
			linearInto(bufs.hidden, bufs.mlpIn, layer.FC1Weight, layer.FC1Bias, dModel, cfg.DecoderFFNDim)
		}
		gelu(bufs.hidden)
		if !bufs.linearGPU(bufs.mlpOut, bufs.hidden, layer.gpuFC2Weight, layer.FC2Bias, cfg.DecoderFFNDim, dModel) {
			linearInto(bufs.mlpOut, bufs.hidden, layer.FC2Weight, layer.FC2Bias, cfg.DecoderFFNDim, dModel)
		}
		for d := range x {
			x[d] += bufs.mlpOut[d]
		}
		decMlpNs += nowNs() - tphase
	}

	// Final LayerNorm
	tphase := nowNs()
	layerNormInto(bufs.normed, x, dec.FinalLNWeight, dec.FinalLNBias, dModel)
	x = bufs.normed

	// LM head: project to vocab (using tied token embedding). Use the dedicated
	// GPU LM-head kernel when the token embedding has been uploaded.
	logits := make([]float32, cfg.VocabSize)
	if dec.lmHeadGPU != nil && bufs.lmHeadGPU(logits, x, dec.lmHeadGPU, cfg.VocabSize, dModel) {
		state.Pos++
		return logits
	}
	if dec.TokenEmbed != nil {
		// Tied-embedding projection x @ TokenEmbed^T over the full vocab; reuse
		// the threaded RVV seqLen=1 path instead of a scalar double loop.
		logits = linearForwardOpt(x[:dModel], dec.TokenEmbed, nil, 1, dModel, cfg.VocabSize)
	}
	decLmNs += nowNs() - tphase

	state.LastToken = tokenID
	state.Pos++
	return logits
}

// causalAttentionSingle computes causal attention for a single query position
// against cached K/V of length seqKV. Optimized with unrolled dot product.
func causalAttentionSingle(q, kCache, vCache []float32, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, dModel)
	scores := make([]float32, seqKV)
	attentionSingleInto(out, q, kCache, vCache, seqKV, numHeads, headDim, scores)
	return out
}

func attentionSingleInto(out, q, kCache, vCache []float32, seqKV, numHeads, headDim int, scores []float32) {
	dModel := numHeads * headDim
	zeroFloat32s(out[:dModel])
	if seqKV <= 0 {
		return
	}
	if len(scores) < seqKV {
		scores = make([]float32, seqKV)
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < numHeads; h++ {
		hOff := h * headDim

		// Compute attention scores with unrolled dot product
		qHead := q[hOff : hOff+headDim]
		for tkv := 0; tkv < seqKV; tkv++ {
			kOff := tkv*dModel + hOff
			scores[tkv] = simdrt.Sdot(qHead, kCache[kOff:kOff+headDim]) * scale
		}

		softmax(scores[:seqKV])

		// Weighted value sum. Use SIMD SAXPY for each value row; headDim is
		// 64 for Whisper large-v3, which maps well to the vector kernel.
		outHead := out[hOff : hOff+headDim]
		for tkv := 0; tkv < seqKV; tkv++ {
			w := scores[tkv]
			if w < 1e-8 {
				continue // skip near-zero weights
			}
			vOff := tkv*dModel + hOff
			simdrt.Saxpy(w, vCache[vOff:vOff+headDim], outHead)
		}
	}
}

// crossAttentionHeadMajor is attentionSingleInto specialized for the cross
// attention, where K/V are precomputed once and re-read for every decoded
// token. kHead/vHead are head-major ([numHeads][seqKV][headDim]) so each head's
// frames are contiguous — the [seqKV,dModel] cache layout otherwise forces a
// stride-dModel cache miss on every frame (the decode's dominant cost).
func crossAttentionHeadMajor(out, q, kHead, vHead []float32, seqKV, numHeads, headDim int, scores []float32) {
	dModel := numHeads * headDim
	zeroFloat32s(out[:dModel])
	if seqKV <= 0 {
		return
	}
	if len(scores) < seqKV {
		scores = make([]float32, seqKV)
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		qHead := q[hOff : hOff+headDim]
		base := h * seqKV * headDim // contiguous block for this head
		for tkv := 0; tkv < seqKV; tkv++ {
			ko := base + tkv*headDim
			scores[tkv] = simdrt.Sdot(qHead, kHead[ko:ko+headDim]) * scale
		}
		softmax(scores[:seqKV])
		outHead := out[hOff : hOff+headDim]
		for tkv := 0; tkv < seqKV; tkv++ {
			w := scores[tkv]
			if w < 1e-8 {
				continue
			}
			vo := base + tkv*headDim
			simdrt.Saxpy(w, vHead[vo:vo+headDim], outHead)
		}
	}
}

// toHeadMajor reorders a [seqKV, numHeads*headDim] cache into head-major
// [numHeads, seqKV, headDim] for contiguous per-head access.
func toHeadMajor(src []float32, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, seqKV*dModel)
	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		base := h * seqKV * headDim
		for t := 0; t < seqKV; t++ {
			copy(out[base+t*headDim:base+t*headDim+headDim], src[t*dModel+hOff:t*dModel+hOff+headDim])
		}
	}
	return out
}
