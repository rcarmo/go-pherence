package whisper

import "math"

// Decoder implements the Whisper token decoder:
// Token embedding + positional embedding → N decoder layers (self-attn + cross-attn + MLP).
type Decoder struct {
	cfg Config

	// Token embeddings
	TokenEmbed []float32 // [vocabSize, dModel]
	PosEmbed   []float32 // [maxDecoderLength, dModel]

	// Decoder layers
	Layers []DecoderLayer

	// Final LayerNorm
	FinalLNWeight []float32
	FinalLNBias   []float32
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
}

// DecoderState holds cached KV for incremental decoding.
type DecoderState struct {
	// Self-attention KV cache per layer: [layer][pos * dModel]
	SelfKCache [][]float32
	SelfVCache [][]float32

	// Cross-attention KV (computed once from encoder output)
	CrossK [][]float32 // [layer][encLen * dModel]
	CrossV [][]float32 // [layer][encLen * dModel]

	Pos int // Current token position
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
	}

	// Pre-allocate self-attention KV caches
	for l := 0; l < numLayers; l++ {
		state.SelfKCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
		state.SelfVCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
	}

	// Pre-compute cross-attention K/V from encoder output (done once)
	for l := 0; l < numLayers; l++ {
		layer := &dec.Layers[l]
		state.CrossK[l] = linearForwardOpt(encoderOutput, layer.CrossKWeight, layer.CrossKBias, encLen, dModel, dModel)
		state.CrossV[l] = linearForwardOpt(encoderOutput, layer.CrossVWeight, layer.CrossVBias, encLen, dModel, dModel)
	}

	return state
}

// ForwardToken runs one decoder step for a single token.
// Returns logits [vocabSize].
func (dec *Decoder) ForwardToken(tokenID int, state *DecoderState) []float32 {
	cfg := dec.cfg
	dModel := cfg.DecoderDModel
	pos := state.Pos

	// Token embedding + positional embedding
	x := make([]float32, dModel)
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
	encLen := len(state.CrossK[0]) / dModel

	for l := range dec.Layers {
		layer := &dec.Layers[l]

		// --- Causal self-attention ---
		normed := layerNorm(x, layer.SelfAttnLNWeight, layer.SelfAttnLNBias, 1, dModel)

		q := linearForwardOpt(normed, layer.SelfQWeight, layer.SelfQBias, 1, dModel, dModel)
		k := linearForwardOpt(normed, layer.SelfKWeight, layer.SelfKBias, 1, dModel, dModel)
		v := linearForwardOpt(normed, layer.SelfVWeight, layer.SelfVBias, 1, dModel, dModel)

		// Append to KV cache
		state.SelfKCache[l] = append(state.SelfKCache[l], k...)
		state.SelfVCache[l] = append(state.SelfVCache[l], v...)

		// Causal attention: query attends to all cached positions (0..pos)
		seqKV := pos + 1
		selfOut := causalAttentionSingle(q, state.SelfKCache[l], state.SelfVCache[l], seqKV, numHeads, headDim)

		projected := linearForwardOpt(selfOut, layer.SelfOWeight, layer.SelfOBias, 1, dModel, dModel)
		for d := range x {
			x[d] += projected[d]
		}

		// --- Cross-attention ---
		normed = layerNorm(x, layer.CrossAttnLNWeight, layer.CrossAttnLNBias, 1, dModel)
		crossQ := linearForwardOpt(normed, layer.CrossQWeight, layer.CrossQBias, 1, dModel, dModel)

		// Cross-attention: Q from decoder, K/V from encoder (full, non-causal)
		crossOut := fullAttention(crossQ, state.CrossK[l], state.CrossV[l], 1, encLen, numHeads, headDim)
		crossProjected := linearForwardOpt(crossOut, layer.CrossOWeight, layer.CrossOBias, 1, dModel, dModel)
		for d := range x {
			x[d] += crossProjected[d]
		}

		// --- MLP ---
		mlpIn := layerNorm(x, layer.MLPLNWeight, layer.MLPLNBias, 1, dModel)
		hidden := linearForwardOpt(mlpIn, layer.FC1Weight, layer.FC1Bias, 1, dModel, cfg.DecoderFFNDim)
		gelu(hidden)
		mlpOut := linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, 1, cfg.DecoderFFNDim, dModel)
		for d := range x {
			x[d] += mlpOut[d]
		}
	}

	// Final LayerNorm
	x = layerNorm(x, dec.FinalLNWeight, dec.FinalLNBias, 1, dModel)

	// LM head: project to vocab (using token embedding transposed)
	logits := make([]float32, cfg.VocabSize)
	if dec.TokenEmbed != nil {
		for v := 0; v < cfg.VocabSize; v++ {
			var dot float32
			for d := 0; d < dModel; d++ {
				dot += x[d] * dec.TokenEmbed[v*dModel+d]
			}
			logits[v] = dot
		}
	}

	state.Pos++
	return logits
}

// causalAttentionSingle computes causal attention for a single query position
// against cached K/V of length seqKV.
func causalAttentionSingle(q, kCache, vCache []float32, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, dModel)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		scores := make([]float32, seqKV)

		for tkv := 0; tkv < seqKV; tkv++ {
			kOff := tkv*dModel + hOff
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[hOff+d] * kCache[kOff+d]
			}
			scores[tkv] = dot * scale
		}

		softmax(scores)

		for tkv := 0; tkv < seqKV; tkv++ {
			vOff := tkv*dModel + hOff
			w := scores[tkv]
			for d := 0; d < headDim; d++ {
				out[hOff+d] += w * vCache[vOff+d]
			}
		}
	}
	return out
}
