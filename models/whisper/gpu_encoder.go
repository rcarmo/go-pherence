package whisper

import (
	"math"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// GPUEncoder wraps a Whisper encoder with GPU-resident weights for fast inference.
type GPUEncoder struct {
	*Encoder
	cfg Config

	// GPU-resident weights (uploaded once)
	conv1W, conv2W *nv.DevBuf
	layers         []gpuEncoderLayer

	ready bool
}

// Ready reports whether this GPU encoder can run GPU-assisted paths.
func (ge *GPUEncoder) Ready() bool { return ge != nil && ge.ready }

type gpuEncoderLayer struct {
	qW, kW, vW, oW  *nv.DevBuf
	fc1W, fc2W      *nv.DevBuf
	attnLNW, mlpLNW *nv.DevBuf
}

// NewGPUEncoder uploads encoder weights to GPU and returns an accelerated encoder.
func NewGPUEncoder(enc *Encoder, cfg Config) *GPUEncoder {
	if !nv.SgemmReady() {
		return &GPUEncoder{Encoder: enc, cfg: cfg, ready: false}
	}

	ge := &GPUEncoder{Encoder: enc, cfg: cfg, ready: true}
	dModel := cfg.EncoderDModel

	// Upload layer weights
	ge.layers = make([]gpuEncoderLayer, cfg.EncoderLayers)
	for i := range ge.layers {
		l := &enc.Layers[i]
		gl := &ge.layers[i]
		gl.qW = gpuWeight(l.QWeight)
		gl.kW = gpuWeight(l.KWeight)
		gl.vW = gpuWeight(l.VWeight)
		gl.oW = gpuWeight(l.OWeight)
		gl.fc1W = gpuWeight(l.FC1Weight)
		gl.fc2W = gpuWeight(l.FC2Weight)
		gl.attnLNW = gpuWeight(l.AttnLNWeight)
		gl.mlpLNW = gpuWeight(l.MLPLNWeight)
	}
	_ = dModel

	return ge
}

// ForwardGPU runs the encoder with GPU-accelerated linear layers.
// Falls back to CPU for operations without GPU kernels.
func (ge *GPUEncoder) ForwardGPU(mel []float32, T int) []float32 {
	if !ge.ready {
		return ge.Encoder.Forward(mel, T)
	}

	cfg := ge.cfg
	dModel := cfg.EncoderDModel

	// Conv stem: CPU by default (small relative to attention), with an opt-in
	// correctness-first CUDA PTX path for end-to-end GPU graph validation.
	h, ok := conv1dForwardGPU(mel, ge.Encoder.Conv1Weight, ge.Encoder.Conv1Bias, cfg.NumMelBins, T, dModel, 1)
	if !ok {
		h = conv1dForward(mel, ge.Encoder.Conv1Weight, ge.Encoder.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	}
	T1 := T
	gelu(h)
	T2 := (T1+2*1-3)/2 + 1
	if h2, ok := conv1dForwardGPU(h, ge.Encoder.Conv2Weight, ge.Encoder.Conv2Bias, dModel, T1, dModel, 2); ok {
		h = h2
	} else {
		h = conv1dForward(h, ge.Encoder.Conv2Weight, ge.Encoder.Conv2Bias, dModel, T1, dModel, 3, 2, 1)
	}
	gelu(h)

	// Transpose to [T2, d_model]
	ht := transpose2D(h, dModel, T2)

	// Add positional embeddings
	for t := 0; t < T2 && t < cfg.MaxLength; t++ {
		for d := 0; d < dModel; d++ {
			ht[t*dModel+d] += ge.Encoder.PosEmbed[t*dModel+d]
		}
	}

	// Encoder layers with GPU GEMV
	for i := range ge.layers {
		ht = ge.forwardLayerGPU(i, ht, T2)
	}

	// Final LayerNorm
	if ge.Encoder.FinalLNWeight != nil {
		ht = layerNorm(ht, ge.Encoder.FinalLNWeight, ge.Encoder.FinalLNBias, T2, dModel)
	}
	return ht
}

func (ge *GPUEncoder) forwardLayerGPU(layerIdx int, x []float32, seqLen int) []float32 {
	cfg := ge.cfg
	dModel := cfg.EncoderDModel
	layer := &ge.Encoder.Layers[layerIdx]
	gl := &ge.layers[layerIdx]

	// Pre-attention LayerNorm (CPU — lightweight)
	normed := layerNorm(x, layer.AttnLNWeight, layer.AttnLNBias, seqLen, dModel)

	// Q, K, V projections via GPU SGEMM
	q := gemvGPU(normed, gl.qW, layer.QBias, seqLen, dModel, dModel)
	k := gemvGPU(normed, gl.kW, layer.KBias, seqLen, dModel, dModel)
	v := gemvGPU(normed, gl.vW, layer.VBias, seqLen, dModel, dModel)

	// Full attention: CPU by default. Opt-in correctness-first CUDA/PTX dispatch
	// validates the GPU graph body but is not expected to be faster than the
	// optimized CPU/SIMD path until a tiled attention kernel lands.
	attnOut, ok := fullAttentionGPU(q, k, v, seqLen, cfg.EncoderHeads, cfg.HeadDim)
	if !ok {
		attnOut = fullAttention(q, k, v, seqLen, seqLen, cfg.EncoderHeads, cfg.HeadDim)
	}

	// Output projection via GPU
	projected := gemvGPU(attnOut, gl.oW, layer.OBias, seqLen, dModel, dModel)

	// Residual
	for i := range x {
		projected[i] += x[i]
	}

	// Pre-MLP LayerNorm
	mlpIn := layerNorm(projected, layer.MLPLNWeight, layer.MLPLNBias, seqLen, dModel)

	// MLP via GPU
	ffnDim := cfg.EncoderFFNDim
	hidden := gemvGPU(mlpIn, gl.fc1W, layer.FC1Bias, seqLen, dModel, ffnDim)
	gelu(hidden)
	mlpOut := gemvGPU(hidden, gl.fc2W, layer.FC2Bias, seqLen, ffnDim, dModel)

	// Residual
	for i := range projected {
		mlpOut[i] += projected[i]
	}

	return mlpOut
}

func conv1dForwardGPU(input, weight, bias []float32, inCh, inLen, outCh, stride int) ([]float32, bool) {
	if !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_CONV1D") || !nv.SgemmReady() || stride <= 0 || inCh <= 0 || inLen <= 0 || outCh <= 0 || len(input) < inCh*inLen || len(weight) < outCh*inCh*3 {
		return nil, false
	}
	outLen := (inLen+2*1-3)/stride + 1
	if outLen <= 0 {
		return nil, false
	}
	inBuf, err := nv.Malloc(len(input))
	if err != nil {
		return nil, false
	}
	defer inBuf.Free()
	wBuf, err := nv.Malloc(len(weight))
	if err != nil {
		return nil, false
	}
	defer wBuf.Free()
	outBuf, err := nv.Malloc(outCh * outLen)
	if err != nil {
		return nil, false
	}
	defer outBuf.Free()
	var bBuf *nv.Buffer
	if len(bias) >= outCh {
		bBuf, err = nv.Malloc(outCh)
		if err != nil {
			return nil, false
		}
		defer bBuf.Free()
		if err := bBuf.Upload(bias[:outCh]); err != nil {
			return nil, false
		}
	}
	if err := inBuf.Upload(input[:inCh*inLen]); err != nil {
		return nil, false
	}
	if err := wBuf.Upload(weight[:outCh*inCh*3]); err != nil {
		return nil, false
	}
	if stride == 1 {
		err = nv.WhisperConv1DK3S1Buffer(outBuf, inBuf, wBuf, bBuf, inCh, inLen, outCh, outLen)
	} else if stride == 2 {
		err = nv.WhisperConv1DK3S2Buffer(outBuf, inBuf, wBuf, bBuf, inCh, inLen, outCh, outLen)
	} else {
		return nil, false
	}
	if err != nil {
		return nil, false
	}
	out := make([]float32, outCh*outLen)
	if err := outBuf.Download(out); err != nil {
		return nil, false
	}
	return out, true
}

func fullAttentionGPU(q, k, v []float32, seqLen, numHeads, headDim int) ([]float32, bool) {
	if !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_ATTENTION") || !nv.SgemmReady() || seqLen <= 0 || numHeads <= 0 || headDim <= 0 {
		return nil, false
	}
	n := seqLen * numHeads * headDim
	if len(q) < n || len(k) < n || len(v) < n {
		return nil, false
	}
	qBuf, err := nv.Malloc(n)
	if err != nil {
		return nil, false
	}
	defer qBuf.Free()
	kBuf, err := nv.Malloc(n)
	if err != nil {
		return nil, false
	}
	defer kBuf.Free()
	vBuf, err := nv.Malloc(n)
	if err != nil {
		return nil, false
	}
	defer vBuf.Free()
	outBuf, err := nv.Malloc(n)
	if err != nil {
		return nil, false
	}
	defer outBuf.Free()
	if err := qBuf.Upload(q[:n]); err != nil {
		return nil, false
	}
	if err := kBuf.Upload(k[:n]); err != nil {
		return nil, false
	}
	if err := vBuf.Upload(v[:n]); err != nil {
		return nil, false
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := nv.WhisperAttentionFullBuffer(outBuf, qBuf, kBuf, vBuf, seqLen, seqLen, numHeads, headDim, scale); err != nil {
		return nil, false
	}
	out := make([]float32, n)
	if err := outBuf.Download(out); err != nil {
		return nil, false
	}
	return out, true
}

func gpuWeight(data []float32) *nv.DevBuf {
	b := nv.NewDevBufFrom(data)
	_ = b.ToGPU()
	return b
}

// gemvGPU computes batched GEMV using GPU SGEMM when profitable.
// x: [seqLen, inDim], W: DevBuf[outDim * inDim] in HF [outDim,inDim] layout, bias: [outDim]
// Returns [seqLen * outDim].
func gemvGPU(x []float32, W *nv.DevBuf, bias []float32, seqLen, inDim, outDim int) []float32 {
	// Use GPU SGEMM for batched matmul (seqLen >= 8). Keep W resident on
	// device; only upload the per-chunk activation X^T and download C.
	if seqLen >= 8 && W != nil && nv.SgemmReady() {
		wGPU := W.GPUPtr()
		if wGPU != nil {
			// out[seq,out] = x[seq,in] @ W[out,in]^T.
			// Compute C[outDim,seqLen] = W[outDim,inDim] * X^T[inDim,seqLen].
			xT := make([]float32, inDim*seqLen)
			for s := 0; s < seqLen; s++ {
				for d := 0; d < inDim; d++ {
					xT[d*seqLen+s] = x[s*inDim+d]
				}
			}
			dX, err := nv.Malloc(len(xT))
			if err == nil {
				defer dX.Free()
				dC, err := nv.Malloc(outDim * seqLen)
				if err == nil {
					defer dC.Free()
					if err = dX.Upload(xT); err == nil {
						if err = nv.Sgemm(outDim, seqLen, inDim, 1.0, wGPU, dX, dC); err == nil {
							nv.Sync()
							result := make([]float32, outDim*seqLen)
							if err = dC.Download(result); err == nil {
								out := make([]float32, seqLen*outDim)
								for s := 0; s < seqLen; s++ {
									for o := 0; o < outDim; o++ {
										out[s*outDim+o] = result[o*seqLen+s]
									}
								}
								if bias != nil {
									for s := 0; s < seqLen; s++ {
										for o := 0; o < outDim && o < len(bias); o++ {
											out[s*outDim+o] += bias[o]
										}
									}
								}
								return out
							}
						}
					}
				}
			}
		}
	}

	// CPU fallback (uses SIMD Sdot)
	return linearForwardOpt(x, W.Data(), bias, seqLen, inDim, outDim)
}

func init() {
	// Warm up CUDA context on package init if available.
	_ = nv.SgemmReady()
}

// NewDecoderStateGPU pre-computes cross-attention K/V using GPU SGEMM.
func NewDecoderStateGPU(cfg Config, encoderOutput []float32, encLen int, dec *Decoder) *DecoderState {
	dModel := cfg.DecoderDModel
	numLayers := cfg.DecoderLayers

	state := &DecoderState{
		SelfKCache: make([][]float32, numLayers),
		SelfVCache: make([][]float32, numLayers),
		CrossK:     make([][]float32, numLayers),
		CrossV:     make([][]float32, numLayers),
		LastToken:  -1,
		Bufs:       newDecoderBufs(cfg),
	}
	gpuCrossAttn := UseGPUCrossAttention() && nv.SgemmReady()
	if gpuCrossAttn {
		state.CrossKGPU = make([]*nv.DevBuf, numLayers)
		state.CrossVGPU = make([]*nv.DevBuf, numLayers)
	}

	for l := 0; l < numLayers; l++ {
		state.SelfKCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
		state.SelfVCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
	}

	// Pre-compute cross-attention K/V using GPU SGEMM
	for l := 0; l < numLayers; l++ {
		layer := &dec.Layers[l]
		wK := nv.NewDevBufFrom(layer.CrossKWeight)
		wV := nv.NewDevBufFrom(layer.CrossVWeight)
		state.CrossK[l] = gemvGPU(encoderOutput, wK, layer.CrossKBias, encLen, dModel, dModel)
		state.CrossV[l] = gemvGPU(encoderOutput, wV, layer.CrossVBias, encLen, dModel, dModel)
		if gpuCrossAttn {
			state.CrossKGPU[l] = nv.NewDevBufFrom(state.CrossK[l])
			if err := state.CrossKGPU[l].ToGPU(); err != nil {
				state.CrossKGPU[l] = nil
			}
			state.CrossVGPU[l] = nv.NewDevBufFrom(state.CrossV[l])
			if err := state.CrossVGPU[l].ToGPU(); err != nil {
				state.CrossVGPU[l] = nil
			}
		}
	}

	return state
}
