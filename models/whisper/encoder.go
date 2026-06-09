package whisper

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Encoder phase timers (ns), summed across layers. Layers run serially so plain
// accumulators are safe; reset at the start of each Encoder.Forward.
var (
	encLinearNs int64
	encAttnNs   int64
	encOtherNs  int64
)

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
	convStart := time.Now()

	// Conv1: [numMelBins, T] → [d_model, T] with kernel=3, stride=1, padding=1
	h := conv1dForwardFast(mel, enc.Conv1Weight, enc.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	T1 := T // stride=1 preserves length
	gelu(h)

	// Conv2: [d_model, T] → [d_model, T/2] with kernel=3, stride=2, padding=1
	h = conv1dForwardFast(h, enc.Conv2Weight, enc.Conv2Bias, dModel, T1, dModel, 3, 2, 1)
	T2 := (T1+2*1-3)/2 + 1
	gelu(h)

	// Transpose to [T2, d_model] for transformer layers
	ht := transpose2D(h, dModel, T2)

	// Add positional embeddings
	for t := 0; t < T2 && t < cfg.MaxLength; t++ {
		for d := 0; d < dModel; d++ {
			ht[t*dModel+d] += enc.PosEmbed[t*dModel+d]
		}
	}

	// Encoder layers
	encLinearNs, encAttnNs, encOtherNs = 0, 0, 0
	resetF16Timers()
	resetA100Timers()
	convNs := int64(time.Since(convStart))
	for i := range enc.Layers {
		ht = enc.forwardLayer(&enc.Layers[i], ht, T2)
	}
	if os.Getenv("WHISPER_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[enc] convstem=%.1fs linear=%.1fs attn=%.1fs other=%.1fs\n",
			float64(convNs)/1e9, float64(encLinearNs)/1e9, float64(encAttnNs)/1e9, float64(encOtherNs)/1e9)
		if attnF16 {
			fmt.Fprintln(os.Stderr, f16TimingLine())
		}
		if useA100FC1 || useA100FC2 || useA100FFNFused {
			fmt.Fprintln(os.Stderr, a100TimingLine())
		}
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
	t0 := time.Now()
	normed := layerNorm(x, layer.AttnLNWeight, layer.AttnLNBias, seqLen, dModel)
	encOtherNs += int64(time.Since(t0))

	// Q, K, V projections
	t0 = time.Now()
	q := linearForwardOpt(normed, layer.QWeight, layer.QBias, seqLen, dModel, dModel)
	k := linearForwardOpt(normed, layer.KWeight, layer.KBias, seqLen, dModel, dModel)
	v := linearForwardOpt(normed, layer.VWeight, layer.VBias, seqLen, dModel, dModel)
	encLinearNs += int64(time.Since(t0))

	// Full (non-causal) multi-head attention
	t0 = time.Now()
	attnOut := fullAttention(q, k, v, seqLen, seqLen, numHeads, headDim)
	encAttnNs += int64(time.Since(t0))

	// Output projection
	t0 = time.Now()
	projected := linearForwardOpt(attnOut, layer.OWeight, layer.OBias, seqLen, dModel, dModel)
	encLinearNs += int64(time.Since(t0))

	// Residual
	for i := range x {
		projected[i] += x[i]
	}

	// Pre-MLP LayerNorm
	t0 = time.Now()
	mlpIn := layerNorm(projected, layer.MLPLNWeight, layer.MLPLNBias, seqLen, dModel)
	encOtherNs += int64(time.Since(t0))

	// MLP: FC1 → GELU → FC2
	ffnDim := enc.cfg.EncoderFFNDim
	t0 = time.Now()
	if mlpOut, ok := forwardFFNTiled(mlpIn, layer, projected, seqLen, dModel, ffnDim); ok {
		encLinearNs += int64(time.Since(t0))
		return mlpOut
	}
	if mlpOut, ok := forwardA100FFNFused(mlpIn, layer, projected, seqLen, dModel, ffnDim); ok {
		encLinearNs += int64(time.Since(t0))
		return mlpOut
	}
	hidden := linearForwardOpt(mlpIn, layer.FC1Weight, layer.FC1Bias, seqLen, dModel, ffnDim)
	encLinearNs += int64(time.Since(t0))
	t0 = time.Now()
	gelu(hidden)
	encOtherNs += int64(time.Since(t0))
	t0 = time.Now()
	mlpOut := linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, seqLen, ffnDim, dModel)
	encLinearNs += int64(time.Since(t0))

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

// conv1dForwardFast computes the same Conv1D as conv1dForward but reformulates
// it as im2col + GEMM so the heavy matmul runs through the tiled, threaded RVV
// linearForwardOpt path instead of a 4-deep scalar loop. Output stays
// channel-first [outCh, outLen] to match conv1dForward.
func conv1dForwardFast(input, weight, bias []float32, inCh, inLen, outCh, kernel, stride, padding int) []float32 {
	outLen := (inLen+2*padding-kernel)/stride + 1
	if outLen <= 0 {
		return nil
	}
	K := inCh * kernel
	// im2col: patch[outLen, K], patch[j, ic*kernel+k] = input[ic, j*stride+k-pad].
	patch := make([]float32, outLen*K)
	for j := 0; j < outLen; j++ {
		base := j*stride - padding
		pj := j * K
		for ic := 0; ic < inCh; ic++ {
			inBase := ic * inLen
			pc := pj + ic*kernel
			for k := 0; k < kernel; k++ {
				if idx := base + k; idx >= 0 && idx < inLen {
					patch[pc+k] = input[inBase+idx]
				}
			}
		}
	}
	// outRM[outLen, outCh] = patch @ weight^T + bias (weight is [outCh, K]).
	outRM := linearForwardOpt(patch, weight, bias, outLen, K, outCh)
	// Transpose to channel-first [outCh, outLen].
	out := make([]float32, outCh*outLen)
	for j := 0; j < outLen; j++ {
		row := j * outCh
		for oc := 0; oc < outCh; oc++ {
			out[oc*outLen+j] = outRM[row+oc]
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

// fastTanh is a float32 Padé[7/6] approximation of tanh: |err| < ~1e-4 for
// |x| < 4.97 and saturates beyond. Avoids the libm float64 round-trip in the
// per-element GELU hot loop (246M calls per encoder forward).
func fastTanh(x float32) float32 {
	if x > 4.97 {
		return 1
	}
	if x < -4.97 {
		return -1
	}
	x2 := x * x
	a := x * (135135 + x2*(17325+x2*(378+x2)))
	b := 135135 + x2*(62370+x2*(3150+x2*28))
	return a / b
}

// gelu applies GELU activation in-place (tanh approximation), float32 throughout.
func gelu(x []float32) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i, v := range x {
		inner := float32(c) * (v + 0.044715*v*v*v)
		x[i] = 0.5 * v * (1 + fastTanh(inner))
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
//
// Heads are independent (disjoint output columns) and split across
// linearWorkers goroutines. Within a head the work is expressed as two GEMMs
// over packed contiguous [seq, headDim] buffers:
//
//	scores = scale * Qh @ Kh^T   (SgemmNTTo)
//	outH   = softmax(scores) @ Vh (SgemmNNTo)
//
// This replaces ~2.9e9 per-element Sdot/Saxpy calls with batched RVV GEMMs,
// removing the per-call/reduction overhead that dominated the scalar-call form.
func fullAttention(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32 {
	if attnF16 && os.Getenv("WHISPER_FP16_HEAD_BATCH") != "" {
		return fullAttentionF16(q, k, v, seqQ, seqKV, numHeads, headDim)
	}
	dModel := numHeads * headDim
	out := make([]float32, seqQ*dModel)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	nw := linearWorkers
	if nw > numHeads {
		nw = numHeads
	}
	if nw < 1 {
		nw = 1
	}
	// FP16 attention uses heads as the outer parallel unit, but each attention
	// GEMM is still large enough (e.g. 1500x1504x64) to benefit from row
	// fanout. Shape benchmarks show nt=6 is ~5x faster than nt=1, so keep
	// linearWorkers inside the per-head GEMM path. Whole-head batching remains
	// behind WHISPER_FP16_HEAD_BATCH because it regresses wall time today.
	f16GemmWorkers := linearWorkers

	work := func(hStart, hEnd int) {
		qh := make([]float32, seqQ*headDim)
		kh := make([]float32, seqKV*headDim)
		vh := make([]float32, seqKV*headDim)
		scores := make([]float32, seqQ*seqKV)
		outh := make([]float32, seqQ*headDim)

		// int8/fp16 attention scratch (goroutine-local; heads run sequentially here).
		var (
			mq, nk, kpad                           int
			qi8, ki8, si8, vti8, qp, kp, sp, vtp   []int8
			qs, ks, ss, vts, vhT, sPad             []float32
			cqk, cout                              []int32
			qf16, kf16, sf16, vtf16, kpf16, vtpf16 []uint16
			cqkf16                                 []float32
		)
		if attnF16 {
			kpad = (seqKV + 31) &^ 31
			qf16 = make([]uint16, seqQ*headDim)
			kf16 = make([]uint16, kpad*headDim)
			cqkf16 = make([]float32, seqQ*kpad)
			sf16 = make([]uint16, seqQ*kpad)
			vtf16 = make([]uint16, headDim*kpad)
			kpf16 = make([]uint16, kpad*headDim)
			vtpf16 = make([]uint16, headDim*kpad)
		} else if attnInt8 {
			mq = (seqQ + 3) &^ 3
			nk = (seqKV + 3) &^ 3
			kpad = (seqKV + 7) &^ 7
			qi8 = make([]int8, mq*headDim)
			qs = make([]float32, mq)
			ki8 = make([]int8, nk*headDim)
			ks = make([]float32, nk)
			cqk = make([]int32, mq*nk)
			vhT = make([]float32, headDim*kpad)
			vti8 = make([]int8, headDim*kpad)
			vts = make([]float32, headDim)
			sPad = make([]float32, mq*kpad)
			si8 = make([]int8, mq*kpad)
			ss = make([]float32, mq)
			cout = make([]int32, mq*headDim)
			qp = make([]int8, mq*headDim)
			kp = make([]int8, nk*headDim)
			sp = make([]int8, mq*kpad)
			vtp = make([]int8, headDim*kpad)
		}

		for h := hStart; h < hEnd; h++ {
			hOff := h * headDim
			// Pack head-local contiguous Q/K/V.
			for t := 0; t < seqQ; t++ {
				copy(qh[t*headDim:(t+1)*headDim], q[t*dModel+hOff:t*dModel+hOff+headDim])
			}
			for t := 0; t < seqKV; t++ {
				copy(kh[t*headDim:(t+1)*headDim], k[t*dModel+hOff:t*dModel+hOff+headDim])
				copy(vh[t*headDim:(t+1)*headDim], v[t*dModel+hOff:t*dModel+hOff+headDim])
			}

			if attnF16 {
				attnF16Head(scores, outh, qh, kh, vh, seqQ, seqKV, headDim, scale,
					kpad, f16GemmWorkers, qf16, kf16, sf16, vtf16, cqkf16, kpf16, vtpf16)
				for t := 0; t < seqQ; t++ {
					copy(out[t*dModel+hOff:t*dModel+hOff+headDim], outh[t*headDim:(t+1)*headDim])
				}
				continue
			}
			if attnInt8 {
				attnInt8Head(scores, outh, qh, kh, vh, seqQ, seqKV, headDim, scale,
					mq, nk, kpad, qi8, qs, ki8, ks, cqk, qp, kp,
					vhT, vti8, vts, sPad, si8, ss, cout, sp, vtp)
				for t := 0; t < seqQ; t++ {
					copy(out[t*dModel+hOff:t*dModel+hOff+headDim], outh[t*headDim:(t+1)*headDim])
				}
				continue
			}

			// scores = scale * Qh @ Kh^T   [seqQ, seqKV]
			for i := range scores {
				scores[i] = 0
			}
			if !simdrt.SgemmNTTo(scores, qh, kh, seqQ, seqKV, headDim, scale, headDim, headDim, seqKV) {
				attnScalarHead(scores, qh, kh, seqQ, seqKV, headDim, scale)
			}
			// Row softmax.
			for tq := 0; tq < seqQ; tq++ {
				softmax(scores[tq*seqKV : (tq+1)*seqKV])
			}
			// outH = scores @ Vh   [seqQ, headDim]
			for i := range outh {
				outh[i] = 0
			}
			if !simdrt.SgemmNNTo(outh, scores, vh, seqQ, headDim, seqKV, 1.0, seqKV, headDim, headDim) {
				attnScalarAV(outh, scores, vh, seqQ, seqKV, headDim)
			}
			// Scatter back to out[:, head].
			for t := 0; t < seqQ; t++ {
				copy(out[t*dModel+hOff:t*dModel+hOff+headDim], outh[t*headDim:(t+1)*headDim])
			}
		}
	}

	if nw <= 1 {
		work(0, numHeads)
		return out
	}
	chunk := (numHeads + nw - 1) / nw
	var wg sync.WaitGroup
	for hs := 0; hs < numHeads; hs += chunk {
		he := hs + chunk
		if he > numHeads {
			he = numHeads
		}
		wg.Add(1)
		go func(hs, he int) {
			defer wg.Done()
			work(hs, he)
		}(hs, he)
	}
	wg.Wait()
	return out
}

// attnScalarHead is the portable fallback for scores = scale * Qh @ Kh^T.
func attnScalarHead(scores, qh, kh []float32, seqQ, seqKV, headDim int, scale float32) {
	for tq := 0; tq < seqQ; tq++ {
		qo := tq * headDim
		for tkv := 0; tkv < seqKV; tkv++ {
			ko := tkv * headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qh[qo+d] * kh[ko+d]
			}
			scores[tq*seqKV+tkv] = dot * scale
		}
	}
}

// attnScalarAV is the portable fallback for outH = scores @ Vh.
func attnScalarAV(outh, scores, vh []float32, seqQ, seqKV, headDim int) {
	for tq := 0; tq < seqQ; tq++ {
		oo := tq * headDim
		so := tq * seqKV
		for tkv := 0; tkv < seqKV; tkv++ {
			w := scores[so+tkv]
			vo := tkv * headDim
			for d := 0; d < headDim; d++ {
				outh[oo+d] += w * vh[vo+d]
			}
		}
	}
}

// softmax applies softmax in-place.
// fastExpF32 approximates exp(x) in float32 with range reduction x = k*ln2 + r
// and a degree-5 polynomial for exp(r); error ~1e-7. Avoids the per-element
// libm float64 exp in softmax (the encoder attention does ~1.4B exp calls at
// T=1500). softmax args are <= 0 after max-subtraction.
func fastExpF32(x float32) float32 {
	if x < -87.3 {
		return 0
	}
	const invln2 = 1.4426950408889634
	const ln2 = 0.6931471805599453
	v := x * invln2
	kf := float32(int32(v - 0.5)) // x<=0 -> round to nearest via truncation
	r := x - kf*ln2
	er := 1 + r*(1+r*(0.5+r*(0.16666667+r*(0.041666668+r*0.008333334))))
	bits := uint32((int32(kf) + 127) << 23)
	return er * math.Float32frombits(bits)
}

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
		e := fastExpF32(v - max)
		x[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range x {
			x[i] /= sum
		}
	}
}
