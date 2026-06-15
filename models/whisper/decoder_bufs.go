package whisper

import (
	"math"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// decoderBufs holds pre-allocated working buffers for one decoder forward token step.
// This eliminates per-token allocations in the hot path.
type decoderBufs struct {
	dModel     int
	ffnDim     int
	x          []float32
	normed     []float32
	q, k, v    []float32
	selfOut    []float32
	proj       []float32
	crossQ     []float32
	crossOut   []float32
	crossProj  []float32
	mlpIn      []float32
	hidden     []float32
	mlpOut     []float32
	scores     []float32
	gpuX       *nv.DevBuf
	gpuOut     *nv.DevBuf
	gpuAttnOut *nv.DevBuf
	gpuLogits  *nv.DevBuf
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
	// Decode is bandwidth-bound on streaming the full decoder weight set per
	// token; int8 weights cut that traffic ~4x. M is padded 1->4 internally.
	if useInt8 && int8Eligible(inDim, outDim) {
		copy(out, linearForwardInt8(x[:inDim], weight, bias, 1, inDim, outDim))
		return
	}
	for o := 0; o < outDim; o++ {
		wOff := o * inDim
		sum := simdrt.Sdot(x[:inDim], weight[wOff:wOff+inDim])
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

func (b *decoderBufs) attentionGPU(out, q []float32, kBufs, vBufs []*nv.DevBuf, layer, seqLen, numHeads, headDim int) bool {
	if b == nil || layer < 0 || layer >= len(kBufs) || layer >= len(vBufs) || kBufs[layer] == nil || vBufs[layer] == nil || seqLen <= 0 || numHeads <= 0 || headDim <= 0 || !nv.SgemmReady() {
		return false
	}
	dModel := numHeads * headDim
	if len(q) < dModel || len(out) < dModel {
		return false
	}
	if b.gpuX == nil || b.gpuX.Len() < dModel {
		b.gpuX = nv.NewDevBuf(dModel)
	}
	if b.gpuAttnOut == nil || b.gpuAttnOut.Len() < dModel {
		b.gpuAttnOut = nv.NewDevBuf(dModel)
	}
	copy(b.gpuX.Data()[:dModel], q[:dModel])
	b.gpuX.MarkDirty()
	qBuf := b.gpuX.GPUPtr()
	outBuf := b.gpuAttnOut.GPUPtr()
	kBuf := kBufs[layer].GPUPtr()
	vBuf := vBufs[layer].GPUPtr()
	if qBuf == nil || outBuf == nil || kBuf == nil || vBuf == nil {
		return false
	}
	if err := nv.WhisperCrossAttentionBuffer(outBuf, qBuf, kBuf, vBuf, 1, seqLen, numHeads, headDim, float32(1.0/math.Sqrt(float64(headDim)))); err != nil {
		return false
	}
	b.gpuAttnOut.ToCPU()
	copy(out[:dModel], b.gpuAttnOut.Data()[:dModel])
	return true
}

func (b *decoderBufs) linearGPU(out, x []float32, weight *nv.DevBuf, bias []float32, inDim, outDim int) bool {
	if b == nil || weight == nil || inDim <= 0 || outDim <= 0 || len(x) < inDim || len(out) < outDim || !nv.SgemmReady() {
		return false
	}
	if b.gpuX == nil || b.gpuX.Len() < inDim {
		b.gpuX = nv.NewDevBuf(inDim)
	}
	if b.gpuOut == nil || b.gpuOut.Len() < outDim {
		b.gpuOut = nv.NewDevBuf(outDim)
	}
	copy(b.gpuX.Data()[:inDim], x[:inDim])
	b.gpuX.MarkDirty()
	nv.DevGemv(b.gpuOut, b.gpuX, weight, outDim, inDim)
	copy(out[:outDim], b.gpuOut.Data()[:outDim])
	if bias != nil {
		for i := 0; i < outDim && i < len(bias); i++ {
			out[i] += bias[i]
		}
	}
	return true
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
