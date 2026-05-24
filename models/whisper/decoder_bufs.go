package whisper

// decoderBufs holds pre-allocated working buffers for one decoder forward token step.
// This eliminates per-token allocations in the hot path.
type decoderBufs struct {
	dModel    int
	ffnDim    int
	normed    []float32
	q, k, v   []float32
	selfOut   []float32
	proj      []float32
	crossQ    []float32
	crossOut  []float32
	crossProj []float32
	mlpIn     []float32
	hidden    []float32
	mlpOut    []float32
}

func newDecoderBufs(cfg Config) *decoderBufs {
	dModel := cfg.DecoderDModel
	ffnDim := cfg.DecoderFFNDim
	return &decoderBufs{
		dModel:    dModel,
		ffnDim:    ffnDim,
		normed:    make([]float32, dModel),
		q:         make([]float32, dModel),
		k:         make([]float32, dModel),
		v:         make([]float32, dModel),
		selfOut:   make([]float32, dModel),
		proj:      make([]float32, dModel),
		crossQ:    make([]float32, dModel),
		crossOut:  make([]float32, dModel),
		crossProj: make([]float32, dModel),
		mlpIn:     make([]float32, dModel),
		hidden:    make([]float32, ffnDim),
		mlpOut:    make([]float32, dModel),
	}
}

// linearInto computes out = x @ W^T + bias into a pre-allocated buffer.
func linearInto(out, x, weight, bias []float32, inDim, outDim int) {
	for o := 0; o < outDim; o++ {
		wOff := o * inDim
		var sum float32
		d := 0
		for ; d+3 < inDim; d += 4 {
			sum += x[d]*weight[wOff+d] +
				x[d+1]*weight[wOff+d+1] +
				x[d+2]*weight[wOff+d+2] +
				x[d+3]*weight[wOff+d+3]
		}
		for ; d < inDim; d++ {
			sum += x[d] * weight[wOff+d]
		}
		if bias != nil && o < len(bias) {
			sum += bias[o]
		}
		out[o] = sum
	}
}

// layerNormInto applies LayerNorm into a pre-allocated buffer.
func layerNormInto(out, x, weight, bias []float32, dim int) {
	const eps = 1e-5
	var sum float64
	for d := 0; d < dim; d++ {
		sum += float64(x[d])
	}
	mean := sum / float64(dim)
	var varSum float64
	for d := 0; d < dim; d++ {
		diff := float64(x[d]) - mean
		varSum += diff * diff
	}
	invStd := float32(1.0 / sqrtf64(varSum/float64(dim)+eps))
	for d := 0; d < dim; d++ {
		normed := (x[d] - float32(mean)) * invStd
		if weight != nil {
			normed *= weight[d]
		}
		if bias != nil {
			normed += bias[d]
		}
		out[d] = normed
	}
}

func sqrtf64(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method (2 iterations, good for normalized inputs)
	r := x
	r = 0.5 * (r + x/r)
	r = 0.5 * (r + x/r)
	r = 0.5 * (r + x/r)
	return r
}
