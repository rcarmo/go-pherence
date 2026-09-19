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
	graph bool
}

// Ready reports whether this GPU encoder can run GPU-assisted paths.
func (ge *GPUEncoder) Ready() bool { return ge != nil && ge.ready }

// EnableGraph selects the verified fully resident encoder graph for this instance.
func (ge *GPUEncoder) EnableGraph() {
	if ge != nil {
		ge.graph = true
	}
}

// Close releases all persistent device weights owned by the encoder.
func (ge *GPUEncoder) Close() {
	if ge == nil {
		return
	}
	free := func(buf *nv.DevBuf) {
		if buf != nil {
			buf.Free()
		}
	}
	free(ge.conv1W)
	free(ge.conv2W)
	for i := range ge.layers {
		layer := &ge.layers[i]
		free(layer.qW)
		free(layer.kW)
		free(layer.vW)
		free(layer.oW)
		free(layer.fc1W)
		free(layer.fc2W)
		free(layer.attnLNW)
		free(layer.attnLNB)
		free(layer.mlpLNW)
		free(layer.mlpLNB)
		free(layer.qB)
		free(layer.kB)
		free(layer.vB)
		free(layer.oB)
		free(layer.fc1B)
		free(layer.fc2B)
	}
	ge.layers = nil
	ge.ready = false
}

type gpuEncoderLayer struct {
	qW, kW, vW, oW             *nv.DevBuf
	fc1W, fc2W                 *nv.DevBuf
	attnLNW, attnLNB           *nv.DevBuf
	mlpLNW, mlpLNB             *nv.DevBuf
	qB, kB, vB, oB, fc1B, fc2B *nv.DevBuf
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
		gl.attnLNB = gpuVector(l.AttnLNBias)
		gl.mlpLNW = gpuWeight(l.MLPLNWeight)
		gl.mlpLNB = gpuVector(l.MLPLNBias)
		gl.qB = gpuVector(l.QBias)
		gl.kB = gpuVector(l.KBias)
		gl.vB = gpuVector(l.VBias)
		gl.oB = gpuVector(l.OBias)
		gl.fc1B = gpuVector(l.FC1Bias)
		gl.fc2B = gpuVector(l.FC2Bias)
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
	h, ok := conv1dForwardGPU(mel, ge.Encoder.Conv1Weight, ge.Encoder.Conv1Bias, cfg.NumMelBins, T, dModel, 1, ge.graph)
	if !ok {
		h = conv1dForward(mel, ge.Encoder.Conv1Weight, ge.Encoder.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	}
	T1 := T
	gelu(h)
	T2 := (T1+2*1-3)/2 + 1
	if h2, ok := conv1dForwardGPU(h, ge.Encoder.Conv2Weight, ge.Encoder.Conv2Bias, dModel, T1, dModel, 2, ge.graph); ok {
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
	if ge.graph || whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_RESIDENT") {
		if out, ok := ge.forwardLayerGPUResident(layerIdx, x, seqLen); ok {
			return out
		}
	}

	cfg := ge.cfg
	dModel := cfg.EncoderDModel
	layer := &ge.Encoder.Layers[layerIdx]
	gl := &ge.layers[layerIdx]

	// Pre-attention LayerNorm (CPU — lightweight)
	normed := layerNorm(x, layer.AttnLNWeight, layer.AttnLNBias, seqLen, dModel)

	// Q, K, V projections share one activation upload. Keeping the three
	// resident weights independent avoids checkpoint repacking while removing
	// two full [sequence,width] host-to-device transfers per layer.
	q, k, v, ok := threeGemvGPU(normed, gl.qW, gl.kW, gl.vW, layer.QBias, layer.KBias, layer.VBias, seqLen, dModel, dModel)
	if !ok {
		q = gemvGPU(normed, gl.qW, layer.QBias, seqLen, dModel, dModel)
		k = gemvGPU(normed, gl.kW, layer.KBias, seqLen, dModel, dModel)
		v = gemvGPU(normed, gl.vW, layer.VBias, seqLen, dModel, dModel)
	}

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

func (ge *GPUEncoder) forwardLayerGPUResident(layerIdx int, x []float32, seqLen int) ([]float32, bool) {
	cfg := ge.cfg
	dModel := cfg.EncoderDModel
	ffnDim := cfg.EncoderFFNDim
	layer := &ge.layers[layerIdx]
	if seqLen <= 0 || len(x) < seqLen*dModel || !nv.SgemmReady() ||
		layer.qW == nil || layer.kW == nil || layer.vW == nil || layer.oW == nil ||
		layer.fc1W == nil || layer.fc2W == nil || layer.attnLNW == nil || layer.mlpLNW == nil {
		return nil, false
	}
	for _, buf := range []*nv.DevBuf{layer.qW, layer.kW, layer.vW, layer.oW, layer.fc1W, layer.fc2W, layer.attnLNW, layer.mlpLNW} {
		if buf.GPUPtr() == nil {
			return nil, false
		}
	}

	scratch := make([]*nv.DevBuf, 0, 16)
	alloc := func(n int) (*nv.DevBuf, bool) {
		buf, err := nv.NewDevBufGPU(n)
		if err != nil {
			return nil, false
		}
		scratch = append(scratch, buf)
		return buf, true
	}
	defer func() {
		for _, buf := range scratch {
			buf.Free()
		}
	}()
	zeroBias := map[int]*nv.DevBuf{}
	ensureBias := func(bias *nv.DevBuf, cols int) (*nv.DevBuf, bool) {
		if bias != nil && bias.GPUPtr() != nil {
			return bias, true
		}
		if buf := zeroBias[cols]; buf != nil {
			return buf, true
		}
		buf, ok := alloc(cols)
		if !ok {
			return nil, false
		}
		if err := nv.ZeroFloat32Buffer(buf.GPUBuffer(), cols); err != nil {
			return nil, false
		}
		zeroBias[cols] = buf
		return buf, true
	}
	affine := func(out, in, weight, bias *nv.DevBuf, rows, cols int) bool {
		b, ok := ensureBias(bias, cols)
		if !ok {
			return false
		}
		return nv.WhisperRowAffineBuffer(out.GPUBuffer(), in.GPUBuffer(), weight.GPUBuffer(), b.GPUBuffer(), rows, cols) == nil
	}
	project := func(inRow, weight, bias *nv.DevBuf, rows, inDim, outDim int) (*nv.DevBuf, bool) {
		inT, ok := alloc(inDim * rows)
		if !ok {
			return nil, false
		}
		if err := nv.WhisperTransposeBuffer(inT.GPUBuffer(), inRow.GPUBuffer(), rows, inDim); err != nil {
			return nil, false
		}
		outCol, ok := alloc(outDim * rows)
		if !ok {
			return nil, false
		}
		if err := nv.Sgemm(outDim, rows, inDim, 1, weight.GPUPtr(), inT.GPUBuffer(), outCol.GPUBuffer()); err != nil {
			return nil, false
		}
		outRow, ok := alloc(rows * outDim)
		if !ok {
			return nil, false
		}
		if err := nv.WhisperTransposeBuffer(outRow.GPUBuffer(), outCol.GPUBuffer(), outDim, rows); err != nil {
			return nil, false
		}
		if bias != nil && bias.GPUPtr() != nil {
			if err := nv.WhisperRowBiasBuffer(outRow.GPUBuffer(), bias.GPUBuffer(), rows, outDim); err != nil {
				return nil, false
			}
		}
		return outRow, true
	}

	xBuf := nv.NewDevBufFrom(x[:seqLen*dModel])
	if err := xBuf.ToGPU(); err != nil {
		return nil, false
	}
	defer xBuf.Free()

	attnNorm0, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	if err := nv.IdeogramLayerNormNoAffineBuffer(attnNorm0.GPUBuffer(), xBuf.GPUBuffer(), seqLen, dModel, 1e-5); err != nil {
		return nil, false
	}
	attnNorm, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	if !affine(attnNorm, attnNorm0, layer.attnLNW, layer.attnLNB, seqLen, dModel) {
		return nil, false
	}

	qBuf, ok := project(attnNorm, layer.qW, layer.qB, seqLen, dModel, dModel)
	if !ok {
		return nil, false
	}
	kBuf, ok := project(attnNorm, layer.kW, layer.kB, seqLen, dModel, dModel)
	if !ok {
		return nil, false
	}
	vBuf, ok := project(attnNorm, layer.vW, layer.vB, seqLen, dModel, dModel)
	if !ok {
		return nil, false
	}

	attnOut, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	if ge.graph || whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_ATTENTION") {
		scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
		if err := nv.WhisperAttentionFullBuffer(attnOut.GPUBuffer(), qBuf.GPUBuffer(), kBuf.GPUBuffer(), vBuf.GPUBuffer(), seqLen, seqLen, cfg.EncoderHeads, cfg.HeadDim, scale); err != nil {
			return nil, false
		}
	} else {
		// The correctness-first one-thread-per-query PTX attention is slower than
		// the SIMD oracle at Whisper's 1500-token encoder horizon. Keep the rest
		// of the layer resident while crossing only Q/K/V and the attention result
		// until a tiled attention kernel passes the end-to-end speed gate.
		n := seqLen * dModel
		qHost, kHost, vHost := make([]float32, n), make([]float32, n), make([]float32, n)
		if err := qBuf.GPUBuffer().Download(qHost); err != nil {
			return nil, false
		}
		if err := kBuf.GPUBuffer().Download(kHost); err != nil {
			return nil, false
		}
		if err := vBuf.GPUBuffer().Download(vHost); err != nil {
			return nil, false
		}
		attnHost := fullAttention(qHost, kHost, vHost, seqLen, seqLen, cfg.EncoderHeads, cfg.HeadDim)
		if err := attnOut.GPUBuffer().Upload(attnHost); err != nil {
			return nil, false
		}
	}

	projected, ok := project(attnOut, layer.oW, layer.oB, seqLen, dModel, dModel)
	if !ok {
		return nil, false
	}
	resid0, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	nv.DevAdd(resid0, projected, xBuf)

	mlpNorm0, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	if err := nv.IdeogramLayerNormNoAffineBuffer(mlpNorm0.GPUBuffer(), resid0.GPUBuffer(), seqLen, dModel, 1e-5); err != nil {
		return nil, false
	}
	mlpNorm, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	if !affine(mlpNorm, mlpNorm0, layer.mlpLNW, layer.mlpLNB, seqLen, dModel) {
		return nil, false
	}

	hidden, ok := project(mlpNorm, layer.fc1W, layer.fc1B, seqLen, dModel, ffnDim)
	if !ok {
		return nil, false
	}
	// Whisper specifies erf GELU. The resident kernel uses a 1.5e-7 maximum-error
	// erf approximation and is gated by the real-checkpoint cumulative parity test.
	nv.DevGELUErf(hidden, seqLen*ffnDim)
	mlpOut, ok := project(hidden, layer.fc2W, layer.fc2B, seqLen, ffnDim, dModel)
	if !ok {
		return nil, false
	}
	outBuf, ok := alloc(seqLen * dModel)
	if !ok {
		return nil, false
	}
	nv.DevAdd(outBuf, mlpOut, resid0)
	nv.Sync()

	out := make([]float32, seqLen*dModel)
	if err := outBuf.GPUBuffer().Download(out); err != nil {
		return nil, false
	}
	return out, true
}

func conv1dForwardGPU(input, weight, bias []float32, inCh, inLen, outCh, stride int, force ...bool) ([]float32, bool) {
	forced := len(force) > 0 && force[0]
	if (!forced && !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_CONV1D")) || !nv.SgemmReady() || stride <= 0 || inCh <= 0 || inLen <= 0 || outCh <= 0 || len(input) < inCh*inLen || len(weight) < outCh*inCh*3 {
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

func gpuVector(data []float32) *nv.DevBuf {
	if len(data) == 0 {
		return nil
	}
	return gpuWeight(data)
}

// threeGemvGPU computes three projections from the same row-major input with a
// single transposed activation upload. Outputs stay API-compatible row-major.
func threeGemvGPU(x []float32, w0, w1, w2 *nv.DevBuf, b0, b1, b2 []float32, seqLen, inDim, outDim int) ([]float32, []float32, []float32, bool) {
	if seqLen < 8 || inDim <= 0 || outDim <= 0 || len(x) < seqLen*inDim || !nv.SgemmReady() || w0 == nil || w1 == nil || w2 == nil {
		return nil, nil, nil, false
	}
	weights := []*nv.DevBuf{w0, w1, w2}
	biases := [][]float32{b0, b1, b2}
	for _, weight := range weights {
		if weight.GPUPtr() == nil {
			return nil, nil, nil, false
		}
	}
	xT := make([]float32, inDim*seqLen)
	for row := 0; row < seqLen; row++ {
		for col := 0; col < inDim; col++ {
			xT[col*seqLen+row] = x[row*inDim+col]
		}
	}
	dX, err := nv.Malloc(len(xT))
	if err != nil {
		return nil, nil, nil, false
	}
	defer dX.Free()
	if err := dX.Upload(xT); err != nil {
		return nil, nil, nil, false
	}
	deviceOut := make([]*nv.Buffer, 3)
	for i := range deviceOut {
		deviceOut[i], err = nv.Malloc(outDim * seqLen)
		if err != nil {
			for j := 0; j < i; j++ {
				deviceOut[j].Free()
			}
			return nil, nil, nil, false
		}
	}
	defer func() {
		for _, buf := range deviceOut {
			buf.Free()
		}
	}()
	for i, weight := range weights {
		if err := nv.Sgemm(outDim, seqLen, inDim, 1, weight.GPUPtr(), dX, deviceOut[i]); err != nil {
			return nil, nil, nil, false
		}
	}
	nv.Sync()
	outputs := make([][]float32, 3)
	for i := range outputs {
		columnMajor := make([]float32, outDim*seqLen)
		if err := deviceOut[i].Download(columnMajor); err != nil {
			return nil, nil, nil, false
		}
		out := make([]float32, seqLen*outDim)
		for row := 0; row < seqLen; row++ {
			for col := 0; col < outDim; col++ {
				value := columnMajor[col*seqLen+row]
				if col < len(biases[i]) {
					value += biases[i][col]
				}
				out[row*outDim+col] = value
			}
		}
		outputs[i] = out
	}
	return outputs[0], outputs[1], outputs[2], true
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
		CrossKHead: make([][]float32, numLayers),
		CrossVHead: make([][]float32, numLayers),
		LastToken:  -1,
		Bufs:       newDecoderBufs(cfg),
	}
	gpuCrossKV := UseGPUCrossKV() && nv.SgemmReady()
	gpuCrossAttn := UseGPUCrossAttention() && nv.SgemmReady()
	if gpuCrossAttn {
		state.CrossKGPU = make([]*nv.DevBuf, numLayers)
		state.CrossVGPU = make([]*nv.DevBuf, numLayers)
	}

	for l := 0; l < numLayers; l++ {
		state.SelfKCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
		state.SelfVCache[l] = make([]float32, 0, cfg.MaxDecoderLength*dModel)
	}

	// Pre-compute cross-attention K/V. Keep CPU/SIMD as the default oracle;
	// use GPU SGEMM only when the cross-K/V graph path is explicitly enabled
	// and available. Per-token GPU cross-attention remains separately gated.
	for l := 0; l < numLayers; l++ {
		layer := &dec.Layers[l]
		if gpuCrossKV {
			wK := nv.NewDevBufFrom(layer.CrossKWeight)
			wV := nv.NewDevBufFrom(layer.CrossVWeight)
			state.CrossK[l] = gemvGPU(encoderOutput, wK, layer.CrossKBias, encLen, dModel, dModel)
			state.CrossV[l] = gemvGPU(encoderOutput, wV, layer.CrossVBias, encLen, dModel, dModel)
		} else {
			state.CrossK[l] = linearForwardOpt(encoderOutput, layer.CrossKWeight, layer.CrossKBias, encLen, dModel, dModel)
			state.CrossV[l] = linearForwardOpt(encoderOutput, layer.CrossVWeight, layer.CrossVBias, encLen, dModel, dModel)
		}
		state.CrossKHead[l] = toHeadMajor(state.CrossK[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
		state.CrossVHead[l] = toHeadMajor(state.CrossV[l], encLen, cfg.DecoderHeads, cfg.HeadDim)
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
