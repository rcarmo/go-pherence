package whisper

import (
	"math"
	"runtime"
	"sync"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// AdvanceTokens feeds prompt tokens that need no logits. On the packed-Q5 CPU
// decoder it runs them as one batch: each Q5 weight row and each cross
// attention head's K/V are read once for all rows instead of once per token.
// Every row still goes through exactly the functions and per-element order of
// sequential AdvanceToken (LayerNorm, SdotQ5_0 rows, causal attention over
// its own prefix, cross heads, GELU, residual adds), so the KV cache, Pos and
// all later logits are bit-identical. Other configurations (F32 weights,
// observers, NVIDIA paths) use the sequential loop.
func (dec *Decoder) AdvanceTokens(tokens []int, state *DecoderState) {
	if len(tokens) < 2 || !dec.batchPrefillEligible(state) {
		for _, token := range tokens {
			dec.AdvanceToken(token, state)
		}
		return
	}
	dec.prefillQ5(tokens, state)
}

func (dec *Decoder) batchPrefillEligible(state *DecoderState) bool {
	if dec.q5Layers == nil || len(dec.q5Layers) != len(dec.Layers) || state.CrossAttentionObserver != nil ||
		nv.SgemmReady() || whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_SELF_ATTN") || len(state.CrossKHead) != len(dec.Layers) {
		return false
	}
	for _, p := range dec.q5Layers {
		if p.selfQ == nil || p.selfK == nil || p.selfV == nil || p.selfO == nil || p.crossQ == nil || p.crossO == nil || p.fc1 == nil || p.fc2 == nil {
			return false
		}
	}
	return true
}

// linearQ5Batch is linearQ5Into for n rows: out[i*outDim+o] = SdotQ5_0(x_i, row o) + bias[o].
func linearQ5Batch(out, x []float32, raw []byte, bias []float32, inDim, outDim, n int) {
	rowBytes := inDim / 32 * simdrt.Q5_0BlockBytes
	_, _, _ = out[:n*outDim], x[:n*inDim], raw[:outDim*rowBytes]
	rows := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			w := raw[o*rowBytes : (o+1)*rowBytes]
			for i := 0; i < n; i++ {
				sum := simdrt.SdotQ5_0(x[i*inDim:(i+1)*inDim], w)
				if bias != nil && o < len(bias) {
					sum += bias[o]
				}
				out[i*outDim+o] = sum
			}
		}
	}
	parallelRange(outDim, max(1, min(min(linearWorkers, runtime.GOMAXPROCS(0)), outDim/256)), rows)
}

func parallelRange(n, workers int, fn func(lo, hi int)) {
	if workers <= 1 || n <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); fn(lo, hi) }(lo, min(lo+chunk, n))
	}
	wg.Wait()
}

func (dec *Decoder) prefillQ5(tokens []int, state *DecoderState) {
	cfg := dec.cfg
	d, f, n := cfg.DecoderDModel, cfg.DecoderFFNDim, len(tokens)
	numHeads, headDim := cfg.DecoderHeads, cfg.HeadDim
	pos0 := state.Pos
	if state.Bufs == nil {
		state.Bufs = newDecoderBufs(cfg)
	}
	encLen := 0
	if len(state.CrossK) > 0 {
		encLen = len(state.CrossK[0]) / d
	}
	x := make([]float32, n*d)
	normed := make([]float32, n*d)
	q := make([]float32, n*d)
	k := make([]float32, n*d)
	v := make([]float32, n*d)
	att := make([]float32, n*d)
	proj := make([]float32, n*d)
	hidden := make([]float32, n*f)
	for i, tokenID := range tokens {
		row := x[i*d : (i+1)*d]
		if tokenID >= 0 && tokenID < cfg.VocabSize && dec.TokenEmbed != nil {
			copy(row, dec.TokenEmbed[tokenID*d:(tokenID+1)*d])
		}
		if pos := pos0 + i; dec.PosEmbed != nil && pos < cfg.MaxDecoderLength {
			for j := 0; j < d; j++ {
				row[j] += dec.PosEmbed[pos*d+j]
			}
		}
	}
	workers := max(1, min(linearWorkers, runtime.GOMAXPROCS(0)))
	norm := func(dst []float32, w, b []float32) {
		for i := 0; i < n; i++ {
			layerNormInto(dst[i*d:(i+1)*d], x[i*d:(i+1)*d], w, b, d)
		}
	}
	residual := func(add []float32) {
		for i := range x {
			x[i] += add[i]
		}
	}
	for l := range dec.Layers {
		layer, p := &dec.Layers[l], dec.q5Layers[l]

		tphase := nowNs()
		norm(normed, layer.SelfAttnLNWeight, layer.SelfAttnLNBias)
		linearQ5Batch(q, normed, p.selfQ, layer.SelfQBias, d, d, n)
		linearQ5Batch(k, normed, p.selfK, layer.SelfKBias, d, d, n)
		linearQ5Batch(v, normed, p.selfV, layer.SelfVBias, d, d, n)
		state.SelfKCache[l] = append(state.SelfKCache[l], k...)
		state.SelfVCache[l] = append(state.SelfVCache[l], v...)
		kc, vc := state.SelfKCache[l], state.SelfVCache[l]
		parallelRange(n, workers, func(lo, hi int) {
			scores := make([]float32, pos0+hi)
			for i := lo; i < hi; i++ {
				attentionSingleInto(att[i*d:(i+1)*d], q[i*d:(i+1)*d], kc, vc, pos0+i+1, numHeads, headDim, scores)
			}
		})
		linearQ5Batch(proj, att, p.selfO, layer.SelfOBias, d, d, n)
		residual(proj)
		decSelfNs += nowNs() - tphase

		tphase = nowNs()
		norm(normed, layer.CrossAttnLNWeight, layer.CrossAttnLNBias)
		linearQ5Batch(q, normed, p.crossQ, layer.CrossQBias, d, d, n)
		dec.crossAttentionBatch(att, q, state, l, encLen, pos0, n, workers)
		linearQ5Batch(proj, att, p.crossO, layer.CrossOBias, d, d, n)
		residual(proj)
		decCrossNs += nowNs() - tphase

		tphase = nowNs()
		norm(normed, layer.MLPLNWeight, layer.MLPLNBias)
		linearQ5Batch(hidden, normed, p.fc1, layer.FC1Bias, d, f, n)
		for i := 0; i < n; i++ {
			if state.tanhGELU {
				geluOriginalTanh(hidden[i*f : (i+1)*f])
			} else {
				gelu(hidden[i*f : (i+1)*f])
			}
		}
		linearQ5Batch(proj, hidden, p.fc2, layer.FC2Bias, f, d, n)
		residual(proj)
		decMlpNs += nowNs() - tphase
	}
	copy(state.Bufs.x, x[(n-1)*d:])
	state.LastToken = tokens[n-1]
	state.Pos += n
}

// crossAttentionBatch reproduces forwardToken's packed-Q5 cross attention for
// n rows. With several workers each worker owns the same head range as
// crossAttentionHeadMajorParallel and walks all rows, so a head's K/V stay hot.
func (dec *Decoder) crossAttentionBatch(out, q []float32, state *DecoderState, l, encLen, pos0, n, workers int) {
	cfg := dec.cfg
	d, numHeads, headDim := cfg.DecoderDModel, cfg.DecoderHeads, cfg.HeadDim
	kHead, vHead := state.CrossKHead[l], state.CrossVHead[l]
	hw := min(workers, numHeads)
	if hw <= 1 || encLen <= 0 {
		scores := make([]float32, max(encLen, 1))
		for i := 0; i < n; i++ {
			crossAttentionHeadMajorPadded(out[i*d:(i+1)*d], q[i*d:(i+1)*d], kHead, vHead, encLen, numHeads, headDim, scores, l, pos0+i, nil, state.crossPadKeys)
		}
		return
	}
	zeroFloat32s(out[:n*d])
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	chunk := (numHeads + hw - 1) / hw
	var wg sync.WaitGroup
	for lo := 0; lo < numHeads; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			scores := make([]float32, encLen)
			for i := 0; i < n; i++ {
				crossAttentionHeads(out[i*d:(i+1)*d], q[i*d:(i+1)*d], kHead, vHead, encLen, headDim, scores, l, pos0+i, nil, state.crossPadKeys, scale, lo, hi)
			}
		}(lo, min(lo+chunk, numHeads))
	}
	wg.Wait()
}
