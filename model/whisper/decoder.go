package whisper

import (
	"context"
	"fmt"
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
// CrossAttentionObserver receives one normalised attention row for one decoder
// layer/head at the token position just consumed by ForwardToken. The values are
// transient and must be copied by the caller. It is a diagnostic/word-alignment
// hook; nil preserves the hot path and GPU cross-attention cannot be observed.
type CrossAttentionObserver func(layer, head, tokenPosition int, weights []float32)

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

	Pos       int // Current token position
	LastToken int // Last token fed into ForwardToken, or -1 before prompt
	// Private original-decoder compatibility (explicit option only): number of
	// virtual zero cross-attention keys and the original tanh-form GELU.
	crossPadKeys int
	tanhGELU     bool
	Bufs         *decoderBufs // Reusable buffers (nil = allocate per call)
	// CrossAttentionObserver is invoked synchronously only on the CPU
	// cross-attention path. Callers must not mutate or retain weights.
	CrossAttentionObserver CrossAttentionObserver
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
	state, _ := newDecoderStateContext(nil, cfg, encoderOutput, encLen, dec)
	return state
}

// NewDecoderStateContext checks cancellation between allocations, projections
// and head-major conversions. It has the same validated-input and exclusive
// execution preconditions as NewDecoderState. No partial state escapes on error;
// a running projection or reorder finishes before cancellation is reported.
func NewDecoderStateContext(ctx context.Context, cfg Config, encoderOutput []float32, encLen int, dec *Decoder) (*DecoderState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return newDecoderStateContext(ctx, cfg, encoderOutput, encLen, dec)
}

func newDecoderStateContext(ctx context.Context, cfg Config, encoderOutput []float32, encLen int, dec *Decoder) (*DecoderState, error) {
	if err := speechContextErr(ctx); err != nil {
		return nil, err
	}
	dModel := cfg.DecoderDModel
	numLayers := cfg.DecoderLayers

	state, err := newDecoderSelfStateContext(ctx, cfg)
	if err != nil {
		return nil, err
	}
	state.CrossK = make([][]float32, numLayers)
	state.CrossV = make([][]float32, numLayers)
	state.CrossKHead = make([][]float32, numLayers)
	state.CrossVHead = make([][]float32, numLayers)

	// Pre-compute cross-attention K/V from encoder output (done once)
	// Use GPU SGEMM if available for this large batched matmul
	for l := 0; l < numLayers; l++ {
		if err := speechContextErr(ctx); err != nil {
			return nil, err
		}
		layer := &dec.Layers[l]
		state.CrossK[l] = linearForwardOpt(encoderOutput, layer.CrossKWeight, layer.CrossKBias, encLen, dModel, dModel)
		if err := speechContextErr(ctx); err != nil {
			return nil, err
		}
		state.CrossV[l] = linearForwardOpt(encoderOutput, layer.CrossVWeight, layer.CrossVBias, encLen, dModel, dModel)
		if err := speechContextErr(ctx); err != nil {
			return nil, err
		}
		// Reorder once to head-major so each decoded token reads each head's
		// frames contiguously instead of stride-dModel (the decode bottleneck).
		state.CrossKHead[l] = toHeadMajor(state.CrossK[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
		if err := speechContextErr(ctx); err != nil {
			return nil, err
		}
		state.CrossVHead[l] = toHeadMajor(state.CrossV[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
	}

	if err := speechContextErr(ctx); err != nil {
		return nil, err
	}
	return state, nil
}

func newDecoderSelfStateContext(ctx context.Context, cfg Config) (*DecoderState, error) {
	if err := speechContextErr(ctx); err != nil {
		return nil, err
	}
	s := &DecoderState{SelfKCache: make([][]float32, cfg.DecoderLayers), SelfVCache: make([][]float32, cfg.DecoderLayers), LastToken: -1, Bufs: newDecoderBufs(cfg)}
	for l := 0; l < cfg.DecoderLayers; l++ {
		if err := speechContextErr(ctx); err != nil {
			return nil, err
		}
		s.SelfKCache[l] = make([]float32, 0, cfg.MaxDecoderLength*cfg.DecoderDModel)
		s.SelfVCache[l] = make([]float32, 0, cfg.MaxDecoderLength*cfg.DecoderDModel)
	}
	if err := speechContextErr(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// Private same-window alignment seam. The checked caller owns both states,
// excludes concurrent use and never mutates cross-KV after construction. Only
// immutable CPU cross-KV is shared; fresh self-KV, observer, position and scratch
// are separate. Copy outer slice headers to keep state metadata independent.
// No GPU lifetime or public borrowed-storage contract is introduced.
func newAlignmentDecoderStateContext(ctx context.Context, cfg Config, encLen int, source *DecoderState) (*DecoderState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("alignment state: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil || encLen < 1 || encLen > cfg.MaxLength || cfg.DecoderLayers < 1 || cfg.DecoderDModel < 1 || len(source.CrossKGPU) != 0 || len(source.CrossVGPU) != 0 || len(source.CrossK) != cfg.DecoderLayers || len(source.CrossV) != cfg.DecoderLayers || len(source.CrossKHead) != cfg.DecoderLayers || len(source.CrossVHead) != cfg.DecoderLayers {
		return nil, fmt.Errorf("alignment state: CPU cross-KV geometry required")
	}
	for l := 0; l < cfg.DecoderLayers; l++ {
		n := encLen * cfg.DecoderDModel
		if len(source.CrossK[l]) != n || len(source.CrossV[l]) != n || len(source.CrossKHead[l]) != n || len(source.CrossVHead[l]) != n {
			return nil, fmt.Errorf("alignment state: cross-KV extent mismatch")
		}
	}
	s, err := newDecoderSelfStateContext(ctx, cfg)
	if err != nil {
		return nil, err
	}
	s.CrossK = append([][]float32(nil), source.CrossK...)
	s.CrossV = append([][]float32(nil), source.CrossV...)
	s.CrossKHead = append([][]float32(nil), source.CrossKHead...)
	s.CrossVHead = append([][]float32(nil), source.CrossVHead...)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s, nil
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
		if state.CrossAttentionObserver != nil || state.crossPadKeys > 0 || !bufs.attentionGPU(bufs.crossOut, bufs.crossQ, state.CrossKGPU, state.CrossVGPU, l, encLen, numHeads, headDim) {
			// Alignment explicitly observes the same CPU probabilities used for
			// this attention result; it never combines hidden GPU output with a
			// separately reconstructed diagnostic row.
			crossAttentionHeadMajorPadded(bufs.crossOut, bufs.crossQ, state.CrossKHead[l], state.CrossVHead[l], encLen, numHeads, headDim, bufs.scores, l, pos, state.CrossAttentionObserver, state.crossPadKeys)
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
		if state.tanhGELU {
			geluOriginalTanh(bufs.hidden)
		} else {
			gelu(bufs.hidden)
		}
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

		// Compute attention scores with SIMD dot products. Sdotx4 covers four
		// time rows at once while preserving the scalar tail path.
		qHead := q[hOff : hOff+headDim]
		tkv := 0
		for ; tkv+3 < seqKV; tkv += 4 {
			kOff := tkv*dModel + hOff
			d0, d1, d2, d3, ok := simdrt.Sdotx4(qHead, kCache[kOff:], dModel)
			if !ok {
				break
			}
			scores[tkv+0] = d0 * scale
			scores[tkv+1] = d1 * scale
			scores[tkv+2] = d2 * scale
			scores[tkv+3] = d3 * scale
		}
		for ; tkv < seqKV; tkv++ {
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
	crossAttentionHeadMajorObserved(out, q, kHead, vHead, seqKV, numHeads, headDim, scores, 0, 0, nil)
}

func crossAttentionHeadMajorObserved(out, q, kHead, vHead []float32, seqKV, numHeads, headDim int, scores []float32, layer, position int, observe CrossAttentionObserver) {
	crossAttentionHeadMajorPadded(out, q, kHead, vHead, seqKV, numHeads, headDim, scores, layer, position, observe, 0)
}

// crossAttentionHeadMajorPadded adds padKeys virtual zero keys/values. Their
// score is exactly zero and they contribute only to the softmax denominator,
// matching the original's unmasked zero-padded flash-attention extent. The
// observer sees the real-key probabilities. padKeys==0 is the unchanged path.
func crossAttentionHeadMajorPadded(out, q, kHead, vHead []float32, seqKV, numHeads, headDim int, scores []float32, layer, position int, observe CrossAttentionObserver, padKeys int) {
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
		tkv := 0
		for ; tkv+3 < seqKV; tkv += 4 {
			ko := base + tkv*headDim
			d0, d1, d2, d3, ok := simdrt.Sdotx4(qHead, kHead[ko:], headDim)
			if !ok {
				break
			}
			scores[tkv+0] = d0 * scale
			scores[tkv+1] = d1 * scale
			scores[tkv+2] = d2 * scale
			scores[tkv+3] = d3 * scale
		}
		for ; tkv < seqKV; tkv++ {
			ko := base + tkv*headDim
			scores[tkv] = simdrt.Sdot(qHead, kHead[ko:ko+headDim]) * scale
		}
		if padKeys > 0 {
			softmaxPadded(scores[:seqKV], padKeys)
		} else {
			softmax(scores[:seqKV])
		}
		if observe != nil {
			observe(layer, h, position, scores[:seqKV])
		}
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
