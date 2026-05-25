package whisper

import "math"

// Encoder implements the Whisper audio encoder:
// Conv1D stem → sinusoidal positional encoding → N transformer encoder layers.
type Encoder struct {
	cfg Config

	// Conv stem weights
	Conv1Weight []float32 // [d_model, numMelBins, 3] flattened
	Conv1Bias   []float32 // [d_model]
	Conv2Weight []float32 // [d_model, d_model, 3] flattened
	Conv2Bias   []float32 // [d_model]

	// Positional embedding (sinusoidal, pre-computed)
	PosEmbed []float32 // [maxLength, d_model] flattened

	// Encoder layers
	Layers []EncoderLayer

	// Final LayerNorm (after all encoder layers)
	FinalLNWeight []float32
	FinalLNBias   []float32
}

// EncoderLayer holds weights for one Whisper encoder transformer layer.
type EncoderLayer struct {
	// Self-attention
	AttnLNWeight []float32 // LayerNorm weight [d_model]
	AttnLNBias   []float32 // LayerNorm bias [d_model]
	QWeight      []float32 // [d_model, d_model]
	QBias        []float32 // [d_model]
	KWeight      []float32 // [d_model, d_model]
	KBias        []float32 // Whisper K has no bias in some versions
	VWeight      []float32 // [d_model, d_model]
	VBias        []float32 // [d_model]
	OWeight      []float32 // [d_model, d_model]
	OBias        []float32 // [d_model]

	// MLP
	MLPLNWeight []float32 // LayerNorm weight [d_model]
	MLPLNBias   []float32 // LayerNorm bias [d_model]
	FC1Weight   []float32 // [ffn_dim, d_model]
	FC1Bias     []float32 // [ffn_dim]
	FC2Weight   []float32 // [d_model, ffn_dim]
	FC2Bias     []float32 // [d_model]
}

// NewEncoder creates an Encoder with allocated layers and pre-computed positional embeddings.
func NewEncoder(cfg Config) *Encoder {
	enc := &Encoder{
		cfg:    cfg,
		Layers: make([]EncoderLayer, cfg.EncoderLayers),
	}
	enc.PosEmbed = sinusoidalPositionEmbedding(cfg.MaxLength, cfg.EncoderDModel)
	return enc
}

// Forward runs the encoder on mel spectrogram features.
// mel: [numMelBins, T] flattened as [numMelBins * T] (channel-first)
// Returns encoder hidden states: [T', d_model] flattened.
func (enc *Encoder) Forward(mel []float32, T int) []float32 {
	cfg := enc.cfg
	dModel := cfg.EncoderDModel

	// Conv1: [numMelBins, T] → [d_model, T] with kernel=3, stride=1, padding=1
	h := conv1dForward(mel, enc.Conv1Weight, enc.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	T1 := T // stride=1 preserves length
	gelu(h)

	// Conv2: [d_model, T] → [d_model, T/2] with kernel=3, stride=2, padding=1
	h = conv1dForward(h, enc.Conv2Weight, enc.Conv2Bias, dModel, T1, dModel, 3, 2, 1)
	T2 := (T1+2*1-3)/2 + 1

	// Transpose to [T2, d_model] for transformer layers
	ht := transpose2D(h, dModel, T2)

	// Add positional embeddings
	for t := 0; t < T2 && t < cfg.MaxLength; t++ {
		for d := 0; d < dModel; d++ {
			ht[t*dModel+d] += enc.PosEmbed[t*dModel+d]
		}
	}

	// Encoder layers
	for i := range enc.Layers {
		ht = enc.forwardLayer(&enc.Layers[i], ht, T2)
	}

	// Final LayerNorm
	if enc.FinalLNWeight != nil {
		ht = layerNorm(ht, enc.FinalLNWeight, enc.FinalLNBias, T2, cfg.EncoderDModel)
	}
	return ht // [T2 * d_model]
}

// forwardLayer runs one encoder transformer layer (full self-attention + MLP).
func (enc *Encoder) forwardLayer(layer *EncoderLayer, x []float32, seqLen int) []float32 {
	dModel := enc.cfg.EncoderDModel
	numHeads := enc.cfg.EncoderHeads
	headDim := enc.cfg.HeadDim

	// Pre-attention LayerNorm
	normed := layerNorm(x, layer.AttnLNWeight, layer.AttnLNBias, seqLen, dModel)

	// Q, K, V projections
	q := linearForwardOpt(normed, layer.QWeight, layer.QBias, seqLen, dModel, dModel)
	k := linearForwardOpt(normed, layer.KWeight, layer.KBias, seqLen, dModel, dModel)
	v := linearForwardOpt(normed, layer.VWeight, layer.VBias, seqLen, dModel, dModel)

	// Full (non-causal) multi-head attention
	attnOut := fullAttention(q, k, v, seqLen, seqLen, numHeads, headDim)

	// Output projection
	projected := linearForwardOpt(attnOut, layer.OWeight, layer.OBias, seqLen, dModel, dModel)

	// Residual
	for i := range x {
		projected[i] += x[i]
	}

	// Pre-MLP LayerNorm
	mlpIn := layerNorm(projected, layer.MLPLNWeight, layer.MLPLNBias, seqLen, dModel)

	// MLP: FC1 → GELU → FC2
	ffnDim := enc.cfg.EncoderFFNDim
	hidden := linearForwardOpt(mlpIn, layer.FC1Weight, layer.FC1Bias, seqLen, dModel, ffnDim)
	gelu(hidden)
	mlpOut := linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, seqLen, ffnDim, dModel)

	// Residual
	for i := range projected {
		mlpOut[i] += projected[i]
	}

	return mlpOut
}

// --- Helper functions ---

// sinusoidalPositionEmbedding computes position embeddings for [maxLen, dModel].
func sinusoidalPositionEmbedding(maxLen, dModel int) []float32 {
	pe := make([]float32, maxLen*dModel)
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dModel; i += 2 {
			freq := math.Pow(10000, float64(i)/float64(dModel))
			pe[pos*dModel+i] = float32(math.Sin(float64(pos) / freq))
			if i+1 < dModel {
				pe[pos*dModel+i+1] = float32(math.Cos(float64(pos) / freq))
			}
		}
	}
	return pe
}

// conv1dForward computes Conv1D on flat channel-first input.
func conv1dForward(input, weight, bias []float32, inCh, inLen, outCh, kernel, stride, padding int) []float32 {
	outLen := (inLen+2*padding-kernel)/stride + 1
	if outLen <= 0 {
		return nil
	}
	out := make([]float32, outCh*outLen)
	for oc := 0; oc < outCh; oc++ {
		for j := 0; j < outLen; j++ {
			var sum float32
			base := j*stride - padding
			for ic := 0; ic < inCh; ic++ {
				wOff := (oc*inCh + ic) * kernel
				for k := 0; k < kernel; k++ {
					idx := base + k
					if idx >= 0 && idx < inLen {
						sum += input[ic*inLen+idx] * weight[wOff+k]
					}
				}
			}
			if bias != nil && oc < len(bias) {
				sum += bias[oc]
			}
			out[oc*outLen+j] = sum
		}
	}
	return out
}

// transpose2D transposes [channels, length] to [length, channels].
func transpose2D(data []float32, channels, length int) []float32 {
	out := make([]float32, channels*length)
	for c := 0; c < channels; c++ {
		for t := 0; t < length; t++ {
			out[t*channels+c] = data[c*length+t]
		}
	}
	return out
}

// gelu applies GELU activation in-place (approximation).
func gelu(x []float32) {
	for i, v := range x {
		// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
		x3 := v * v * v
		inner := float32(math.Sqrt(2/math.Pi)) * (v + 0.044715*x3)
		x[i] = 0.5 * v * (1 + float32(math.Tanh(float64(inner))))
	}
}

// layerNorm applies LayerNorm over the last dimension.
// x: [seqLen, dim] flattened.
func layerNorm(x, weight, bias []float32, seqLen, dim int) []float32 {
	out := make([]float32, seqLen*dim)
	const eps = 1e-5
	for t := 0; t < seqLen; t++ {
		off := t * dim
		// Mean
		var sum float64
		for d := 0; d < dim; d++ {
			sum += float64(x[off+d])
		}
		mean := sum / float64(dim)
		// Variance
		var varSum float64
		for d := 0; d < dim; d++ {
			diff := float64(x[off+d]) - mean
			varSum += diff * diff
		}
		invStd := float32(1.0 / math.Sqrt(varSum/float64(dim)+eps))
		for d := 0; d < dim; d++ {
			normed := (x[off+d] - float32(mean)) * invStd
			if weight != nil {
				normed *= weight[d]
			}
			if bias != nil {
				normed += bias[d]
			}
			out[off+d] = normed
		}
	}
	return out
}

// linearForward computes out = x @ W^T + bias.
// x: [seqLen, inDim], W: [outDim, inDim], bias: [outDim]
// Returns [seqLen, outDim].
func linearForwardScalar(x, weight, bias []float32, seqLen, inDim, outDim int) []float32 {
	out := make([]float32, seqLen*outDim)
	for t := 0; t < seqLen; t++ {
		xOff := t * inDim
		oOff := t * outDim
		for o := 0; o < outDim; o++ {
			var sum float32
			wOff := o * inDim
			for d := 0; d < inDim; d++ {
				sum += x[xOff+d] * weight[wOff+d]
			}
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[oOff+o] = sum
		}
	}
	return out
}

// fullAttention computes non-causal multi-head attention.
// q, k, v: [seqQ * dModel] or [seqKV * dModel]
// Returns [seqQ * dModel].
func fullAttention(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, seqQ*dModel)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		for tq := 0; tq < seqQ; tq++ {
			// Compute attention scores for this query position
			scores := make([]float32, seqKV)
			qOff := tq*dModel + hOff

			for tkv := 0; tkv < seqKV; tkv++ {
				kOff := tkv*dModel + hOff
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[qOff+d] * k[kOff+d]
				}
				scores[tkv] = dot * scale
			}

			// Softmax
			softmax(scores)

			// Weighted sum of values
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
	return out
}

// softmax applies softmax in-place.
func softmax(x []float32) {
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
