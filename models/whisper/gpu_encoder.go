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

	// Conv stem still on CPU (small relative to attention)
	h := conv1dForward(mel, ge.Encoder.Conv1Weight, ge.Encoder.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	T1 := T
	gelu(h)
	h = conv1dForward(h, ge.Encoder.Conv2Weight, ge.Encoder.Conv2Bias, dModel, T1, dModel, 3, 2, 1)
	T2 := (T1+2*1-3)/2 + 1
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

	// Full attention (CPU for now — GPU attention kernel needs sequence-level parallelism)
	attnOut := fullAttention(q, k, v, seqLen, seqLen, cfg.EncoderHeads, cfg.HeadDim)

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

// transpose2DGPU is a placeholder for GPU-accelerated transpose.
func transpose2DGPU(data []float32, channels, length int) []float32 {
	return transpose2D(data, channels, length)
}

func init() {
	// Warm up CUDA context on package init if available
	if nv.SgemmReady() {
		_ = math.Pi // prevent unused import
	}
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
	}

	return state
}
