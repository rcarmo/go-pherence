package model

// CPU prefill batching.
//
// The sequential CPU decode loop in generatePrepared processes prompt tokens
// one at a time, so every weight matrix is read once per prompt token. Prefill
// restructures the prompt pass as "for each layer, process all B prompt tokens
// together", turning the seven per-layer projections (Q/K/V/O/gate/up/down)
// into batched GEMMs that read each weight matrix once for all B tokens.
//
// Numerics are kept bit-identical to the sequential path: the batched GEMM
// produces the same per-output dot product as the per-token GEMV, and every
// sequence-dependent step (RMSNorm, RoPE, QK/V-norm, causal attention, KV
// append, activation, residual ordering, BF16 truncation) is replicated
// per token in the same order. The eligibility gate keeps prefill on the
// validated subset and falls back to the sequential loop otherwise.

import (
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
	"github.com/rcarmo/go-pherence/tensor"
)

// prefillCPUEligible reports whether the batched CPU prefill can faithfully
// reproduce the sequential decode body for this model and batch size.
func (m *LlamaModel) prefillCPUEligible(B int) bool {
	if m == nil || B < 2 {
		return false
	}
	if os.Getenv("GO_PHERENCE_DISABLE_CPU_PREFILL") == "1" {
		return false
	}
	// Diagnostics rely on the per-op/per-token sequential trace; never prefill
	// while any trace or override hook is installed.
	if debugOpHook != nil || debugLayerHook != nil || debugLogitsHook != nil ||
		debugCPUHiddenInOverrideHook != nil || debugCPUPerLayerInputsOverrideHook != nil ||
		debugCPUMLPInputOverrideHook != nil {
		return false
	}
	cfg := m.Config
	if cfg.HiddenSize <= 0 || cfg.NumLayers <= 0 || len(m.Layers) < cfg.NumLayers {
		return false
	}
	// Gemma4 per-layer input gating is not reproduced by the batched body.
	if cfg.ModelType == "gemma4_text" || m.PerLayerModelProj != nil {
		return false
	}
	if cfg.NumExperts > 0 {
		return false
	}
	for i := 0; i < cfg.NumLayers; i++ {
		L := &m.Layers[i]
		if L.IsMoE || !L.HasKV || L.PLIGate != nil {
			return false
		}
		// INT4 on-the-fly weights have no batched CPU kernel yet.
		if L.QWq != nil || L.KWq != nil || L.VWq != nil || L.OWq != nil ||
			L.GateWq != nil || L.UpWq != nil || L.DownWq != nil {
			return false
		}
		if (L.QWm == nil && L.QW == nil) || (L.KWm == nil && L.KW == nil) ||
			(L.OWm == nil && L.OW == nil) || (L.GateWm == nil && L.GateW == nil) ||
			(L.UpWm == nil && L.UpW == nil) || (L.DownWm == nil && L.DownW == nil) {
			return false
		}
		if !cfg.AttentionKEqV && L.VWm == nil && L.VW == nil {
			return false
		}
		if cfg.AttentionKEqV && L.VWm == nil && L.VW == nil && L.KWm == nil && L.KW == nil {
			return false
		}
		if L.InputNorm == nil || L.PostNorm == nil {
			return false
		}
	}
	return true
}

// projBatch computes out[B, outDim] = x[B, inDim] @ W^T for all B rows in one
// batched kernel, mirroring m.mv / mlx.Gemv numerics exactly. Returns false on
// malformed inputs so the caller can abort prefill and fall back.
func (m *LlamaModel) projBatch(out, x []float32, B int, dense *tensor.Tensor, mlxw *mlx.QuantWeight, inDim, outDim int) bool {
	if B <= 0 || inDim <= 0 || outDim <= 0 {
		return false
	}
	if len(out) < B*outDim || len(x) < B*inDim {
		return false
	}
	if mlxw != nil {
		return mlx.Gemm(out[:B*outDim], x[:B*inDim], B, mlxw)
	}
	if dense == nil {
		return false
	}
	wd := dense.Data()
	if m.Large {
		// gemvNT layout: W is [outDim, inDim] row-major. Parallel across rows.
		return simd.GemmRowsParallel(out[:B*outDim], x[:B*inDim], wd, B, outDim, inDim)
	}
	// gemv layout: W is [inDim, outDim] pre-transposed; SgemmNN accumulates,
	// so the destination must be zeroed first (matches the freshly-allocated
	// per-token output buffers in the sequential path). Parallel across columns.
	dst := out[:B*outDim]
	for i := range dst {
		dst[i] = 0
	}
	return simd.SgemmNNParallelTo(dst, x[:B*inDim], wd, B, outDim, inDim, 1.0, inDim, outDim, outDim)
}

// layerInterFor returns the MLP intermediate width for a layer, mirroring the
// per-layer sizing logic of the sequential decode loop.
func (m *LlamaModel) layerInterFor(layer *LlamaLayer) int {
	inter := m.Config.Intermediate
	switch {
	case layer.GateWm != nil && layer.GateWm.OutDim > 0:
		return layer.GateWm.OutDim
	case layer.GateW != nil:
		s := layer.GateW.Shape()
		if len(s) >= 2 {
			if m.Large {
				return s[0]
			}
			return s[1]
		}
		if len(s) == 1 && s[0] > 0 {
			return s[0]
		}
	}
	return inter
}

// prefillCPU runs the prompt through the model with batched projections and
// fills kvCacheK/kvCacheV for all prompt positions. It returns the hidden state
// for the final prompt token (pre-LM-head) and ok=true on success. On any
// unsupported shape it returns ok=false having left the caches untouched-enough
// for the caller to fall back to the sequential loop (the caller must only use
// this on freshly-allocated caches).
func (m *LlamaModel) prefillCPU(tokenIDs []int, kvCacheK, kvCacheV [][]float32) ([]float32, bool) {
	cfg := m.Config
	B := len(tokenIDs)
	if !m.prefillCPUEligible(B) {
		return nil, false
	}
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	headDim := cfg.HeadDim
	if h <= 0 || numHeads <= 0 || headDim <= 0 {
		return nil, false
	}
	isGemma := cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text"

	batchHidden, okBatchHidden := checkedProduct(B, h)
	if !okBatchHidden {
		return nil, false
	}

	// Embed all prompt tokens: bHidden[B, h].
	bHidden := make([]float32, batchHidden)
	for i, tokID := range tokenIDs {
		if err := m.ScaledTokenEmbeddingInto(bHidden[i*h:(i+1)*h], tokID); err != nil {
			return nil, false
		}
	}

	// Batched scratch reused across layers.
	bResidual := make([]float32, batchHidden)
	bNormed := make([]float32, batchHidden)
	bMlpIn := make([]float32, batchHidden)
	maxQDim, okMaxQ := checkedProduct(numHeads, headDim)
	maxKVDim, okMaxKV := checkedProduct(cfg.NumKVHeads, headDim)
	if !okMaxQ || !okMaxKV {
		return nil, false
	}
	maxInter := cfg.Intermediate
	for l := 0; l < cfg.NumLayers; l++ {
		if d := m.Layers[l].HeadDimLocal; d > headDim {
			q, okQ := checkedProduct(numHeads, d)
			kv, okKV := checkedProduct(cfg.NumKVHeads, d)
			if !okQ || !okKV {
				return nil, false
			}
			if q > maxQDim {
				maxQDim = q
			}
			if kv > maxKVDim {
				maxKVDim = kv
			}
		}
		if li := m.layerInterFor(&m.Layers[l]); li > maxInter {
			maxInter = li
		}
	}
	if maxQDim <= 0 || maxKVDim <= 0 || maxInter <= 0 {
		return nil, false
	}
	batchQ, okBatchQ := checkedProduct(B, maxQDim)
	batchKV, okBatchKV := checkedProduct(B, maxKVDim)
	batchInter, okBatchInter := checkedProduct(B, maxInter)
	if !okBatchQ || !okBatchKV || !okBatchInter {
		return nil, false
	}
	bQ := make([]float32, batchQ)
	bK := make([]float32, batchKV)
	bV := make([]float32, batchKV)
	bAttnOut := make([]float32, batchQ)
	bOOut := make([]float32, batchHidden)
	bGate := make([]float32, batchInter)
	bUp := make([]float32, batchInter)
	bDown := make([]float32, batchHidden)

	eps := float32(cfg.RMSNormEps)
	normFn := rmsNormInPlace
	if isGemma {
		normFn = rmsNormBF16
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		layerHeadDim := headDim
		if layer.HeadDimLocal > 0 {
			layerHeadDim = layer.HeadDimLocal
		}
		layerKVHeads := gemmacfg.LayerKVHeads(cfg, l)
		qDim, okQDim := checkedProduct(numHeads, layerHeadDim)
		layerKVDim, okKVDim := checkedProduct(layerKVHeads, layerHeadDim)
		if qDim <= 0 || layerKVDim <= 0 || !okQDim || !okKVDim {
			return nil, false
		}

		// residual = hidden; normed = RMSNorm(hidden, InputNorm).
		copy(bResidual, bHidden)
		copy(bNormed, bHidden)
		inNorm := layer.InputNorm.Data()
		for b := 0; b < B; b++ {
			row := bNormed[b*h : (b+1)*h]
			if isGemma {
				simd.RMSNormBF16(row, inNorm, eps)
			} else {
				rmsNormInPlace(row, inNorm, eps)
			}
		}

		// Batched Q/K/V projections.
		if !m.projBatch(bQ, bNormed, B, layer.QW, layer.QWm, h, qDim) {
			return nil, false
		}
		if !m.projBatch(bK, bNormed, B, layer.KW, layer.KWm, h, layerKVDim) {
			return nil, false
		}
		// V can alias or be omitted when AttentionKEqV; copy K in either case.
		if cfg.AttentionKEqV && ((layer.KWm != nil && (layer.VWm == nil || layer.VWm == layer.KWm)) || (layer.KW != nil && (layer.VW == nil || layer.VW == layer.KW))) {
			copy(bV[:B*layerKVDim], bK[:B*layerKVDim])
		} else if !m.projBatch(bV, bNormed, B, layer.VW, layer.VWm, h, layerKVDim) {
			return nil, false
		}

		// Per-token attention front-end (BF16, bias, V-norm, QK-norm, RoPE, KV
		// append). This phase mutates q/k/v in place and appends to the cache in
		// sequence order, so it stays serial.
		for b := 0; b < B; b++ {
			pos := b
			q := bQ[b*qDim : (b+1)*qDim]
			k := bK[b*layerKVDim : (b+1)*layerKVDim]
			v := bV[b*layerKVDim : (b+1)*layerKVDim]

			if isGemma {
				simd.ToBF16(q)
				simd.ToBF16(k)
				simd.ToBF16(v)
			}

			if layer.QB != nil {
				simd.VecAdd(q, q, layer.QB.Data())
				if layer.KB != nil {
					simd.VecAdd(k, k, layer.KB.Data())
				}
				if layer.VB != nil {
					simd.VecAdd(v, v, layer.VB.Data())
				}
			}

			// V-norm.
			if cfg.ModelType == "gemma4_text" {
				for head := 0; head < layerKVHeads; head++ {
					simd.RMSNormNoScale(v[head*layerHeadDim:(head+1)*layerHeadDim], eps)
				}
			} else if layer.VNorm != nil {
				vnorm := layer.VNorm.Data()
				for head := 0; head < layerKVHeads; head++ {
					normFn(v[head*layerHeadDim:(head+1)*layerHeadDim], vnorm, eps)
				}
			}

			// QK-norm.
			if layer.QNorm != nil {
				qNorm := layer.QNorm.Data()
				for head := 0; head < numHeads; head++ {
					normFn(q[head*layerHeadDim:(head+1)*layerHeadDim], qNorm, eps)
				}
				if layer.KNorm == nil {
					return nil, false
				}
				kNorm := layer.KNorm.Data()
				for head := 0; head < layerKVHeads; head++ {
					normFn(k[head*layerHeadDim:(head+1)*layerHeadDim], kNorm, eps)
				}
			}

			// RoPE (standard path; Gemma4 dual-rope is excluded by eligibility).
			freqs := m.ensureRoPE(pos)
			applyRoPE(q, freqs, pos, numHeads, layerHeadDim)
			applyRoPE(k, freqs, pos, layerKVHeads, layerHeadDim)

			// KV append (plain caches only).
			kvCacheK[l] = append(kvCacheK[l], k...)
			kvCacheV[l] = append(kvCacheV[l], v...)
		}

		// Causal attention per token. Tokens are independent now that the cache
		// is fully populated, so this runs in parallel across the prompt.
		kCacheL := kvCacheK[l]
		vCacheL := kvCacheV[l]
		scale := float32(1.0 / math.Sqrt(float64(layerHeadDim)))
		attnErr := parallelForTokens(B, func(b int, scores []float32) bool {
			pos := b
			seqLen := pos + 1
			attnSeqLen := seqLen
			attnKVOffset := 0
			if cfg.SlidingWindow > 0 && len(cfg.LayerTypes) > l && cfg.LayerTypes[l] == "sliding_attention" {
				if seqLen > cfg.SlidingWindow {
					attnSeqLen = cfg.SlidingWindow
					attnKVOffset = seqLen - cfg.SlidingWindow
				}
			}
			q := bQ[b*qDim : (b+1)*qDim]
			out := bAttnOut[b*qDim : (b+1)*qDim]
			off, okOff := checkedProduct(attnKVOffset, layerKVDim)
			end, okEnd := checkedProduct(seqLen, layerKVDim)
			if !okOff || !okEnd || end > len(kCacheL) || end > len(vCacheL) || off > end {
				return false
			}
			gqaAttentionScaleSoftcapInto(out, scores[:attnSeqLen], q, kCacheL[off:end], vCacheL[off:end], attnSeqLen, numHeads, layerKVHeads, layerHeadDim, scale, attentionLogitSoftcap(cfg))
			return true
		})
		if !attnErr {
			return nil, false
		}

		// Batched output projection.
		if !m.projBatch(bOOut, bAttnOut, B, layer.OW, layer.OWm, qDim, h) {
			return nil, false
		}

		// Residual + post-attention norm, producing the MLP input.
		postNorm := layer.PostNorm.Data()
		if layer.PreFFNNorm != nil {
			preNorm := layer.PreFFNNorm.Data()
			for b := 0; b < B; b++ {
				o := bOOut[b*h : (b+1)*h]
				rmsNormInPlace(o, postNorm, eps)
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], o)
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				in := bMlpIn[b*h : (b+1)*h]
				copy(in, bHidden[b*h:(b+1)*h])
				if isGemma {
					simd.RMSNormBF16(in, preNorm, eps)
				} else {
					rmsNormInPlace(in, preNorm, eps)
				}
			}
		} else {
			for b := 0; b < B; b++ {
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], bOOut[b*h:(b+1)*h])
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				rmsNormInPlace(bHidden[b*h:(b+1)*h], postNorm, eps)
			}
			copy(bMlpIn, bHidden)
		}

		// Batched MLP.
		layerInter := m.layerInterFor(layer)
		if !m.projBatch(bGate, bMlpIn, B, layer.GateW, layer.GateWm, h, layerInter) {
			return nil, false
		}
		if !m.projBatch(bUp, bMlpIn, B, layer.UpW, layer.UpWm, h, layerInter) {
			return nil, false
		}
		for b := 0; b < B; b++ {
			gate := bGate[b*layerInter : (b+1)*layerInter]
			up := bUp[b*layerInter : (b+1)*layerInter]
			if isGemma {
				simd.ToBF16(gate)
				simd.ToBF16(up)
			}
			if cfg.HiddenAct == "gelu_pytorch_tanh" {
				simd.GELUTanhMul(gate, gate, up)
				if isGemma {
					simd.ToBF16(gate)
				}
			} else {
				simd.VecSiLUMul(gate, gate, up)
			}
		}
		if !m.projBatch(bDown, bGate, B, layer.DownW, layer.DownWm, layerInter, h) {
			return nil, false
		}

		// Down BF16, post-FFN norm, residual, layer scalar, final BF16.
		for b := 0; b < B; b++ {
			down := bDown[b*h : (b+1)*h]
			if isGemma {
				simd.ToBF16(down)
			}
			if layer.PostFFNNorm != nil {
				if isGemma {
					rmsNormBF16(down, layer.PostFFNNorm.Data(), eps)
				} else {
					rmsNormInPlace(down, layer.PostFFNNorm.Data(), eps)
				}
			}
			hid := bHidden[b*h : (b+1)*h]
			simd.VecAdd(hid, bResidual[b*h:(b+1)*h], down)
			if layer.LayerScalar != 1.0 && layer.LayerScalar != 0 {
				simd.VecScale(hid, hid, layer.LayerScalar)
			}
			if isGemma {
				simd.ToBF16(hid)
			}
		}
	}

	lastHidden := make([]float32, h)
	copy(lastHidden, bHidden[(B-1)*h:B*h])
	return lastHidden, true
}

// parallelForTokens runs fn for each token index in [0,B) across GOMAXPROCS
// workers using round-robin distribution (which balances the triangular cost
// of causal attention). Each worker gets its own scratch scores buffer of
// length B. It returns false if any invocation reports failure.
func parallelForTokens(B int, fn func(b int, scores []float32) bool) bool {
	if B <= 0 {
		return true
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers < 1 {
		nWorkers = 1
	}
	if nWorkers > B {
		nWorkers = B
	}
	if nWorkers == 1 {
		scores := make([]float32, B)
		for b := 0; b < B; b++ {
			if !fn(b, scores) {
				return false
			}
		}
		return true
	}
	ok := int32(1)
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			scores := make([]float32, B)
			for b := start; b < B; b += nWorkers {
				if !fn(b, scores) {
					atomic.StoreInt32(&ok, 0)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	return atomic.LoadInt32(&ok) == 1
}
