package speaker

import "math"

// ECAPA implements the ECAPA-TDNN speaker embedding model.
// Architecture: Conv1D blocks with SE-blocks + attentive statistics pooling → embedding.
type ECAPA struct {
	cfg Config

	// Layer 0: initial conv
	Conv0Weight []float32 // [channels[0], numMels, kernelSize]
	Conv0Bias   []float32 // [channels[0]]

	// SE-Res2Net blocks (simplified as Conv1D + SE for now)
	Blocks []ECAPABlock

	// Aggregation layer (concat channel outputs)
	AggWeight []float32 // [channels[-1], sum(channels[:-1]), 1]
	AggBias   []float32

	// Attentive statistics pooling
	PoolAttnWeight []float32 // [attentionDim, channels[-1]]
	PoolAttnBias   []float32
	PoolOutWeight  []float32 // [attentionDim, 1]
	PoolOutBias    []float32

	// Final embedding projection
	EmbedWeight []float32 // [embedDim, channels[-1] * 2]  (*2 for mean+std)
	EmbedBias   []float32
}

// ECAPABlock is one ECAPA-TDNN convolutional block with SE.
type ECAPABlock struct {
	// 1D convolution
	ConvWeight []float32 // [outCh, inCh, kernelSize]
	ConvBias   []float32

	// Batch norm (simplified as scale+shift)
	BNWeight []float32
	BNBias   []float32

	// SE block
	SEDown []float32 // [seBottleneck, channels]
	SEUp   []float32 // [channels, seBottleneck]
}

// NewECAPA creates an ECAPA-TDNN model with the given config.
func NewECAPA(cfg Config) *ECAPA {
	numBlocks := len(cfg.Channels) - 1
	return &ECAPA{
		cfg:    cfg,
		Blocks: make([]ECAPABlock, numBlocks),
	}
}

// Embed computes a speaker embedding from mel spectrogram features.
// mel: [numMels, T] flattened channel-first.
// Returns embedding of size [embedDim].
func (e *ECAPA) Embed(mel []float32, T int) []float32 {
	cfg := e.cfg
	ch0 := cfg.Channels[0]

	// Initial conv
	h := conv1d(mel, e.Conv0Weight, e.Conv0Bias, cfg.NumMels, T, ch0, cfg.KernelSize, 1, cfg.KernelSize/2)
	hLen := T // stride=1 preserves length
	relu(h)

	// Process blocks
	var blockOutputs [][]float32
	blockOutputs = append(blockOutputs, copySlice(h))

	inCh := ch0
	for i, block := range e.Blocks {
		outCh := cfg.Channels[i+1]
		h = conv1d(h, block.ConvWeight, block.ConvBias, inCh, hLen, outCh, cfg.KernelSize, 1, cfg.KernelSize/2)
		relu(h)

		// SE block: global avg pool → fc_down → relu → fc_up → sigmoid → scale
		h = seBlock(h, block.SEDown, block.SEUp, outCh, hLen, cfg.SEBottleneck)

		blockOutputs = append(blockOutputs, copySlice(h))
		inCh = outCh
	}

	// Attentive statistics pooling
	lastCh := cfg.Channels[len(cfg.Channels)-1]
	embedding := attentiveStatPool(h, e.PoolAttnWeight, e.PoolAttnBias, e.PoolOutWeight, e.PoolOutBias, lastCh, hLen, cfg.AttentionDim)

	// Final linear projection to embed dim
	out := linearProject(embedding, e.EmbedWeight, e.EmbedBias, len(embedding), cfg.EmbedDim)

	return out
}

// --- Helpers ---

func conv1d(input, weight, bias []float32, inCh, inLen, outCh, kernel, stride, padding int) []float32 {
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
					if idx >= 0 && idx < inLen && wOff+k < len(weight) {
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

func seBlock(h, seDown, seUp []float32, channels, length, bottleneck int) []float32 {
	if seDown == nil || seUp == nil {
		return h
	}

	// Global average pooling
	avg := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var sum float32
		for t := 0; t < length; t++ {
			sum += h[c*length+t]
		}
		avg[c] = sum / float32(length)
	}

	// FC down
	down := make([]float32, bottleneck)
	for b := 0; b < bottleneck; b++ {
		var sum float32
		for c := 0; c < channels; c++ {
			if b*channels+c < len(seDown) {
				sum += avg[c] * seDown[b*channels+c]
			}
		}
		down[b] = max32(sum, 0) // ReLU
	}

	// FC up + sigmoid
	scale := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var sum float32
		for b := 0; b < bottleneck; b++ {
			if c*bottleneck+b < len(seUp) {
				sum += down[b] * seUp[c*bottleneck+b]
			}
		}
		scale[c] = sigmoid(sum)
	}

	// Scale channels
	for c := 0; c < channels; c++ {
		for t := 0; t < length; t++ {
			h[c*length+t] *= scale[c]
		}
	}
	return h
}

func attentiveStatPool(h, attnW, attnB, outW, outB []float32, channels, length, attnDim int) []float32 {
	if attnW == nil || outW == nil {
		// Fallback: simple mean pooling + std
		mean := make([]float32, channels)
		std := make([]float32, channels)
		for c := 0; c < channels; c++ {
			var sum, sq float64
			for t := 0; t < length; t++ {
				v := float64(h[c*length+t])
				sum += v
				sq += v * v
			}
			mean[c] = float32(sum / float64(length))
			variance := sq/float64(length) - (sum/float64(length))*(sum/float64(length))
			if variance < 0 {
				variance = 0
			}
			std[c] = float32(math.Sqrt(variance))
		}
		return append(mean, std...)
	}

	// Compute attention weights per time step
	weights := make([]float32, length)
	for t := 0; t < length; t++ {
		// Hidden = tanh(W @ h_t + b)
		var maxScore float32
		for a := 0; a < attnDim; a++ {
			var sum float32
			for c := 0; c < channels; c++ {
				if a*channels+c < len(attnW) {
					sum += h[c*length+t] * attnW[a*channels+c]
				}
			}
			if attnB != nil && a < len(attnB) {
				sum += attnB[a]
			}
			sum = float32(math.Tanh(float64(sum)))
			// Score = V @ hidden
			if a < len(outW) {
				maxScore += sum * outW[a]
			}
		}
		if outB != nil && len(outB) > 0 {
			maxScore += outB[0]
		}
		weights[t] = maxScore
	}

	// Softmax over time
	softmaxInPlace(weights)

	// Weighted mean and std
	mean := make([]float32, channels)
	std := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var wMean, wSq float64
		for t := 0; t < length; t++ {
			v := float64(h[c*length+t])
			w := float64(weights[t])
			wMean += v * w
			wSq += v * v * w
		}
		mean[c] = float32(wMean)
		variance := wSq - wMean*wMean
		if variance < 0 {
			variance = 0
		}
		std[c] = float32(math.Sqrt(variance))
	}

	return append(mean, std...)
}

func linearProject(input, weight, bias []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		var sum float32
		for d := 0; d < inDim; d++ {
			if o*inDim+d < len(weight) {
				sum += input[d] * weight[o*inDim+d]
			}
		}
		if bias != nil && o < len(bias) {
			sum += bias[o]
		}
		out[o] = sum
	}
	return out
}

func relu(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func softmaxInPlace(x []float32) {
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

func copySlice(s []float32) []float32 {
	c := make([]float32, len(s))
	copy(c, s)
	return c
}
