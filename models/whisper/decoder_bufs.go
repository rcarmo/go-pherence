package whisper

import (
	"math"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// decoderBufs holds pre-allocated working buffers for one decoder forward token step.
// This eliminates per-token allocations in the hot path.
type decoderBufs struct {
	dModel    int
	ffnDim    int
	x         []float32
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
	scores    []float32
	gpuX      *nv.DevBuf
	gpuLogits *nv.DevBuf
}

func newDecoderBufs(cfg Config) *decoderBufs {
	dModel := cfg.DecoderDModel
	ffnDim := cfg.DecoderFFNDim
	return &decoderBufs{
		dModel:    dModel,
		ffnDim:    ffnDim,
		x:         make([]float32, dModel),
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
		scores:    make([]float32, max(cfg.MaxLength, cfg.MaxDecoderLength)),
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
	invStd := float32(1.0 / math.Sqrt(varSum/float64(dim)+eps))
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

func (b *decoderBufs) lmHeadGPU(logits, x []float32, weight *nv.DevBuf, vocab, h int) bool {
	if b == nil || weight == nil || vocab <= 0 || h <= 0 || len(logits) < vocab || len(x) < h || !nv.SgemmReady() {
		return false
	}
	if b.gpuX == nil || b.gpuX.Len() < h {
		b.gpuX = nv.NewDevBuf(h)
	}
	if b.gpuLogits == nil || b.gpuLogits.Len() < vocab {
		b.gpuLogits = nv.NewDevBuf(vocab)
	}
	copy(b.gpuX.Data()[:h], x[:h])
	b.gpuX.MarkDirty()
	nv.DevLMHead(b.gpuLogits, b.gpuX, weight, vocab, h)
	copy(logits[:vocab], b.gpuLogits.Data()[:vocab])
	return true
}

func zeroFloat32s(x []float32) {
	for i := range x {
		x[i] = 0
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
