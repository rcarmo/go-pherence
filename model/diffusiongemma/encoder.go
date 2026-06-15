package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// EncodePrompt runs the CPU/reference prompt encoder.
//
// This path is intentionally kept available for llama.cpp parity checks and
// golden fixture generation. Production generation should still prefer the GPU
// backend graph; callers must opt into CPUDispatcher explicitly.
func (d CPUDispatcher) EncodePrompt(promptIDs []int, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) ([]EncoderKVLayer, error) {
	return d.EncodePromptWithFP8(promptIDs, weights, ops, buffers, nil, nil, nil)
}

func (d CPUDispatcher) EncodePromptWithFP8(promptIDs []int, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan, fp8 *GPUFP8Model, fp8w *FP8TextWeights, expertCache *ExpertLRUCache) ([]EncoderKVLayer, error) {
	if weights == nil {
		return nil, fmt.Errorf("DiffusionGemma encoder missing weights")
	}
	if len(promptIDs) == 0 {
		return nil, nil
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, fmt.Errorf("DiffusionGemma encoder missing embed_tokens")
	}
	hiddenSize := fp.Globals.EmbedTokens.Shape[1]
	positions := len(promptIDs)

	// Preload resident layer weights for the encoder pass
	if d.ResidentLayerPrefix > 0 {
		// Always use lightweight preload — BF16 path reads raw mmap directly,
		// FP8 path uses GPU projections. Only norms/scalars need F32 decode.
		if err := weights.PreloadLayerRangeLightweight(0, d.ResidentLayerPrefix); err != nil {
			return nil, err
		}
	}

	// Embed prompt tokens with scale
	hiddenElems, ok := checked.MulInt(positions, hiddenSize)
	if !ok {
		return nil, fmt.Errorf("DiffusionGemma encoder hidden size overflow positions=%d hidden=%d", positions, hiddenSize)
	}
	hidden := make([]float32, hiddenElems)
	for i, token := range promptIDs {
		row, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, token)
		if err != nil {
			return nil, err
		}
		if len(shape) != 1 || shape[0] != hiddenSize {
			return nil, fmt.Errorf("DiffusionGemma encoder embed row shape %v want [%d]", shape, hiddenSize)
		}
		if err := decodeFloatRowTo(hidden[i*hiddenSize:(i+1)*hiddenSize], row, dtype); err != nil {
			return nil, err
		}
	}
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range hidden {
		hidden[i] *= embedScale
	}

	// Allocate per-layer KV storage
	numLayers := len(fp.Layers)
	kvLayers := make([]EncoderKVLayer, numLayers)
	residual := make([]float32, len(hidden))
	slidingWindow := buffers.SlidingWindow
	if slidingWindow <= 0 {
		// llama.cpp reads attention.sliding_window from the model metadata; keep
		// the published DiffusionGemma default only as a compatibility fallback.
		slidingWindow = 1024
	}

	// Run each layer with prefetch of next layer's weights
	encoderStarted := time.Now()
	encoderTempDenseStatsStart := ggufTempDenseUploadSnapshot()
	encoderGGUFExpertStatsStart := ggufExpertDispatchStatsSnapshot()
	encoderCPUExpertStatsStart := ggufCPUExpertTimingSnapshot()
	var totalNorm, totalAttention, totalDense, totalMoE, totalOther, totalPrefetchWait time.Duration
	var encoderMoeResult []float32
	var encoderGGUFRouter []float32
	var encoderGGUFNormedRows []float32
	var encoderGGUFTopKIDs []int
	var encoderGGUFTopKVals []float32
	var prefetchDone <-chan struct{}
	var activeTempDenseSession *ggufTempDenseUploadSession
	defer func() {
		if activeTempDenseSession != nil {
			activeTempDenseSession.Close()
		}
	}()
	for layer := 0; layer < numLayers; layer++ {
		// Start prefetching layer+1 weights while this layer computes
		if layer+1 < numLayers {
			prefetchDone = prefetchLayerWeights(weights, fp, layer+1)
		}
		layerStart := time.Now()
		var tNorm, tAttention, tDense, tMoE, tOther time.Duration
		var layerTempDense *ggufTempDenseUploadSession
		tempDense := func() *ggufTempDenseUploadSession {
			if layerTempDense == nil {
				sess := beginGGUFTempDenseUploadSession()
				layerTempDense = sess
				activeTempDenseSession = sess
			}
			return layerTempDense
		}
		lb := fp.Layers[layer]
		lt := ""
		if layer < len(ops.Layers) {
			for _, op := range ops.Layers {
				if op.Layer == layer {
					lt = op.Type
					break
				}
			}
		}

		// input_norm + save residual
		t0Norm := time.Now()
		copy(residual, hidden)
		normW, err := loadFloatVector(weights, lb.InputLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], normW, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma encoder input norm rejected layer %d", layer)
			}
		}
		tNorm += time.Since(t0Norm)

		// attention: get shapes and try BF16 native path first
		t0Attention := time.Now()
		if lb.QProj == nil || lb.KProj == nil || lb.OProj == nil || len(lb.QProj.Shape) != 2 || len(lb.KProj.Shape) != 2 || len(lb.OProj.Shape) != 2 {
			return nil, fmt.Errorf("DiffusionGemma encoder attention bindings missing layer=%d", layer)
		}
		qRows, qCols := lb.QProj.Shape[0], lb.QProj.Shape[1]
		kRows, kCols := lb.KProj.Shape[0], lb.KProj.Shape[1]
		vRows, vCols := kRows, hiddenSize
		if lb.VProj != nil {
			if len(lb.VProj.Shape) != 2 {
				return nil, fmt.Errorf("DiffusionGemma encoder V shape %v layer=%d", lb.VProj.Shape, layer)
			}
			vRows, vCols = lb.VProj.Shape[0], lb.VProj.Shape[1]
		}
		oRows, oCols := lb.OProj.Shape[0], lb.OProj.Shape[1]
		if qCols != hiddenSize || kCols != hiddenSize || vCols != hiddenSize || oRows != hiddenSize || oCols != qRows {
			return nil, fmt.Errorf("DiffusionGemma encoder attention shape mismatch layer=%d Q=[%d,%d] K=[%d,%d] V=[%d,%d] O=[%d,%d] hidden=%d", layer, qRows, qCols, kRows, kCols, vRows, vCols, oRows, oCols, hiddenSize)
		}
		// Try BF16 zero-copy from mmap (avoids F32 decode entirely)
		qBF16, _, err := weights.RawBF16Tensor(lb.QProj.Name)
		if err != nil {
			return nil, err
		}
		kBF16, _, err := weights.RawBF16Tensor(lb.KProj.Name)
		if err != nil {
			return nil, err
		}
		useBF16 := qBF16 != nil && kBF16 != nil
		// Decode F32 fallbacks for any projection that is not served by BF16/FP8.
		var qW, kW, vW []float32
		var vBF16 []uint16
		if fp8 == nil {
			if !useBF16 {
				var err2 error
				qW, _, _, err2 = loadFloatMatrix(weights, lb.QProj)
				if err2 != nil {
					return nil, err2
				}
				kW, _, _, err2 = loadFloatMatrix(weights, lb.KProj)
				if err2 != nil {
					return nil, err2
				}
			}
			if lb.VProj != nil {
				var err2 error
				vBF16, _, err2 = weights.RawBF16Tensor(lb.VProj.Name)
				if err2 != nil {
					return nil, err2
				}
				if vBF16 == nil {
					vW, _, _, err2 = loadFloatMatrix(weights, lb.VProj)
					if err2 != nil {
						return nil, err2
					}
				}
			}
		}
		qNorm, err := loadFloatVector(weights, lb.QNorm)
		if err != nil {
			return nil, err
		}
		kNorm, err := loadFloatVector(weights, lb.KNorm)
		if err != nil {
			return nil, err
		}
		headDim := len(qNorm)
		if headDim <= 0 || len(kNorm) != headDim || qRows <= 0 || kRows <= 0 || vRows <= 0 || qRows%headDim != 0 || kRows%headDim != 0 || vRows != kRows {
			return nil, fmt.Errorf("DiffusionGemma encoder head shape mismatch layer=%d qRows=%d kRows=%d vRows=%d qNorm=%d kNorm=%d", layer, qRows, kRows, vRows, len(qNorm), len(kNorm))
		}
		heads := qRows / headDim
		kvHeads := kRows / headDim
		if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 {
			return nil, fmt.Errorf("DiffusionGemma encoder GQA shape mismatch layer=%d heads=%d kvHeads=%d headDim=%d", layer, heads, kvHeads, headDim)
		}

		qAllLen, okQ := checked.MulInt(positions, qRows)
		kAllLen, okK := checked.MulInt(positions, kRows)
		vAllLen, okV := checked.MulInt(positions, vRows)
		if !okQ || !okK || !okV {
			return nil, fmt.Errorf("DiffusionGemma encoder attention buffer overflow layer=%d positions=%d q=%d k=%d v=%d", layer, positions, qRows, kRows, vRows)
		}
		qAll := make([]float32, qAllLen)
		kAll := make([]float32, kAllLen)
		vAll := make([]float32, vAllLen)

		ropeHalf := headDim / 2
		ropeTheta := 10000.0
		var ropeFactors []float32
		if lt == "full_attention" {
			// llama.cpp: full-attention layers use n_rot_full=headDim plus
			// rope_freqs.weight factors for proportional RoPE. FP8 safetensors
			// omit rope_freqs, so synthesize the same factors from config defaults.
			ropeTheta = 1000000.0
			factors, err := fullAttentionRoPEFactors(weights, fp, headDim)
			if err != nil {
				return nil, err
			}
			ropeFactors = factors
		}
		ropeFreqs := simd.BuildRoPEFreqsWithFactors(positions, ropeHalf, headDim, ropeTheta, ropeFactors)

		qAct := make([]float32, hiddenSize)
		ggufPrefillGPUQKV := fp8 == nil && !useBF16 && gpu.SgemmReady()
		var prefillAttn *GGUFGPUAttentionWeights
		if ggufPrefillGPUQKV {
			oWForCache, oRowsForCache, oColsForCache, err := loadFloatMatrix(weights, lb.OProj)
			if err != nil {
				return nil, err
			}
			if oRowsForCache != hiddenSize || oColsForCache != qRows {
				return nil, fmt.Errorf("encoder GPU O shape mismatch layer=%d O=[%d,%d] hidden=%d qRows=%d", layer, oRowsForCache, oColsForCache, hiddenSize, qRows)
			}
			if d.ggufDenseLayerResident(layer) {
				prefillAttn, err = residentGGUFGPUAttentionWeights(layer, lb, qW, kW, vW, oWForCache, qRows, kRows, vRows, hiddenSize)
				if err != nil {
					return nil, err
				}
			} else {
				tmp := tempDense()
				qBuf, err := tmp.Upload("encoder_attn_q", qW, qRows, hiddenSize)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU Q upload rejected layer=%d: %w", layer, err)
				}
				kBuf, err := tmp.Upload("encoder_attn_k", kW, kRows, hiddenSize)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU K upload rejected layer=%d: %w", layer, err)
				}
				var vBuf *gpu.Buffer
				if lb.VProj != nil {
					vBuf, err = tmp.Upload("encoder_attn_v", vW, vRows, hiddenSize)
					if err != nil {
						return nil, fmt.Errorf("encoder temporary GPU V upload rejected layer=%d: %w", layer, err)
					}
				}
				oBuf, err := tmp.Upload("encoder_attn_o", oWForCache, hiddenSize, qRows)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU O upload rejected layer=%d: %w", layer, err)
				}
				prefillAttn = &GGUFGPUAttentionWeights{Q: qBuf, K: kBuf, V: vBuf, O: oBuf, QRows: qRows, KRows: kRows, VRows: vRows, Hidden: hiddenSize}
			}
			if err := batchedGPUGemmTransposed(qAll, hidden, positions, qRows, hiddenSize, prefillAttn.Q); err != nil {
				return nil, fmt.Errorf("encoder GPU Q GEMM rejected layer=%d: %w", layer, err)
			}
			if err := batchedGPUGemmTransposed(kAll, hidden, positions, kRows, hiddenSize, prefillAttn.K); err != nil {
				return nil, fmt.Errorf("encoder GPU K GEMM rejected layer=%d: %w", layer, err)
			}
			if lb.VProj != nil {
				if prefillAttn.V == nil {
					return nil, fmt.Errorf("encoder GPU V missing layer=%d", layer)
				}
				if err := batchedGPUGemmTransposed(vAll, hidden, positions, vRows, hiddenSize, prefillAttn.V); err != nil {
					return nil, fmt.Errorf("encoder GPU V GEMM rejected layer=%d: %w", layer, err)
				}
			} else {
				copy(vAll, kAll)
			}
		}
		for pos := 0; pos < positions; pos++ {
			h := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
			q := qAll[pos*qRows : (pos+1)*qRows]
			k := kAll[pos*kRows : (pos+1)*kRows]
			v := vAll[pos*vRows : (pos+1)*vRows]
			if fp8 != nil && layer < len(fp8.Layers) {
				fl := &fp8.Layers[layer]
				hIn := h
				if diffusionGemmaFP8DynamicActivationEnabled() {
					hIn = quantizeDynamicTokenRow(qAct, h)
				}
				if err := gpu.GemvFP8E4M3(q, hIn, fl.Q); err != nil {
					return nil, fmt.Errorf("encoder FP8 Q GEMV rejected layer=%d pos=%d: %w", layer, pos, err)
				}
				if err := gpu.GemvFP8E4M3(k, hIn, fl.K); err != nil {
					return nil, fmt.Errorf("encoder FP8 K GEMV rejected layer=%d pos=%d: %w", layer, pos, err)
				}
				if fl.V != nil {
					if err := gpu.GemvFP8E4M3(v, hIn, fl.V); err != nil {
						return nil, fmt.Errorf("encoder FP8 V GEMV rejected layer=%d pos=%d: %w", layer, pos, err)
					}
				} else {
					copy(v, k)
				}
			} else if ggufPrefillGPUQKV {
				// Q/K/V were computed by GGUF GPU GEMMs above.
			} else if useBF16 {
				if !bf16GemvNarrow(q, h, qBF16, qRows, hiddenSize) || !bf16GemvNarrow(k, h, kBF16, kRows, hiddenSize) {
					return nil, fmt.Errorf("encoder BF16 Q/K GEMV rejected layer=%d pos=%d", layer, pos)
				}
				if lb.VProj != nil {
					if vBF16 != nil {
						if !bf16GemvNarrow(v, h, vBF16, vRows, hiddenSize) {
							return nil, fmt.Errorf("encoder BF16 V GEMV rejected layer=%d pos=%d", layer, pos)
						}
					} else if !simd.GemvRowsParallel(v, h, vW, vRows, hiddenSize) {
						return nil, fmt.Errorf("encoder BF16-fallback V GEMV rejected layer=%d pos=%d", layer, pos)
					}
				} else {
					copy(v, k)
				}
			} else {
				if !simd.GemvRowsParallel(q, h, qW, qRows, hiddenSize) || !simd.GemvRowsParallel(k, h, kW, kRows, hiddenSize) {
					return nil, fmt.Errorf("encoder Q/K GEMV rejected layer=%d pos=%d", layer, pos)
				}
				if lb.VProj != nil {
					if !simd.GemvRowsParallel(v, h, vW, vRows, hiddenSize) {
						return nil, fmt.Errorf("encoder V GEMV rejected layer=%d pos=%d", layer, pos)
					}
				} else {
					copy(v, k)
				}
			}
			for hh := 0; hh < heads; hh++ {
				if !simd.RMSNormTo(q[hh*headDim:(hh+1)*headDim], qNorm, 1e-6) {
					return nil, fmt.Errorf("encoder Q RMSNorm rejected layer=%d pos=%d head=%d", layer, pos, hh)
				}
			}
			for hh := 0; hh < kvHeads; hh++ {
				if !simd.RMSNormTo(k[hh*headDim:(hh+1)*headDim], kNorm, 1e-6) {
					return nil, fmt.Errorf("encoder K RMSNorm rejected layer=%d pos=%d head=%d", layer, pos, hh)
				}
				if !simd.RMSNormNoScaleTo(v[hh*headDim:(hh+1)*headDim], 1e-6) {
					return nil, fmt.Errorf("encoder V RMSNorm rejected layer=%d pos=%d head=%d", layer, pos, hh)
				}
			}
			if len(ropeFreqs) > 0 && ropeHalf > 0 {
				simd.ApplyRoPEPartial(q, ropeFreqs, pos, heads, headDim, ropeHalf)
				simd.ApplyRoPEPartial(k, ropeFreqs, pos, kvHeads, headDim, ropeHalf)
			}
		}

		// Save K,V for this layer's encoder KV cache
		kvLayers[layer] = EncoderKVLayer{
			Keys:    append([]float32(nil), kAll...),
			Values:  append([]float32(nil), vAll...),
			SeqLen:  positions,
			KVHeads: kvHeads,
			HeadDim: headDim,
		}

		// Causal self-attention
		group := heads / kvHeads
		var oW []float32
		var oBF16 []uint16
		if fp8 == nil {
			var err2 error
			if useBF16 {
				oBF16, _, err2 = weights.RawBF16Tensor(lb.OProj.Name)
				if err2 != nil {
					return nil, err2
				}
			}
			if !useBF16 || oBF16 == nil {
				oW, _, _, err2 = loadFloatMatrix(weights, lb.OProj)
				if err2 != nil {
					return nil, err2
				}
			}
		}
		attnAll := make([]float32, positions*qRows)
		if ggufPrefillGPUQKV && (lt != "sliding_attention" || positions <= slidingWindow) {
			if err := gpu.F32BatchedCausalGQAAttention(attnAll, qAll, kAll, vAll, positions, heads, kvHeads, headDim, 1.0); err != nil {
				return nil, fmt.Errorf("encoder GPU causal attention rejected layer=%d: %w", layer, err)
			}
		}
		if ggufPrefillGPUQKV && (lt != "sliding_attention" || positions <= slidingWindow) {
			if prefillAttn == nil || prefillAttn.O == nil {
				return nil, fmt.Errorf("encoder GPU O missing layer=%d", layer)
			}
			if err := batchedGPUGemmTransposed(hidden, attnAll, positions, hiddenSize, qRows, prefillAttn.O); err != nil {
				return nil, fmt.Errorf("encoder GPU O GEMM rejected layer=%d: %w", layer, err)
			}
		} else {
			attnCtx := make([]float32, qRows)
			out := make([]float32, hiddenSize)
			scores := make([]float32, positions)
			for pos := 0; pos < positions; pos++ {
				if ggufPrefillGPUQKV && (lt != "sliding_attention" || positions <= slidingWindow) {
					copy(attnCtx, attnAll[pos*qRows:(pos+1)*qRows])
				} else {
					for i := range attnCtx {
						attnCtx[i] = 0
					}
					for hh := 0; hh < heads; hh++ {
						kvh := hh / group
						q := qAll[pos*qRows+hh*headDim : pos*qRows+(hh+1)*headDim]
						for j := 0; j < positions; j++ {
							if j > pos || (lt == "sliding_attention" && pos-j >= slidingWindow) {
								scores[j] = float32(math.Inf(-1)) // llama.cpp prompt mask: causal + SWA clip
							} else {
								scores[j] = dot(q, kAll[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
							}
						}
						softmaxInPlace(scores[:positions])
						dst := attnCtx[hh*headDim : (hh+1)*headDim]
						for j := 0; j < positions; j++ {
							vv := vAll[j*vRows+kvh*headDim : j*vRows+(kvh+1)*headDim]
							for dd := range dst {
								dst[dd] += scores[j] * vv[dd]
							}
						}
					}
				}
				if fp8 != nil && layer < len(fp8.Layers) {
					attnIn := attnCtx
					if diffusionGemmaFP8DynamicActivationEnabled() {
						attnIn = quantizeDynamicTokenRow(qAct, attnCtx)
					}
					if err := gpu.GemvFP8E4M3(out, attnIn, fp8.Layers[layer].O); err != nil {
						return nil, fmt.Errorf("encoder FP8 O GEMV rejected layer=%d pos=%d: %w", layer, pos, err)
					}
				} else if useBF16 {
					if oBF16 != nil {
						if !bf16GemvNarrow(out, attnCtx, oBF16, hiddenSize, qRows) {
							return nil, fmt.Errorf("encoder BF16 O GEMV rejected layer=%d pos=%d", layer, pos)
						}
					} else if !simd.GemvRowsParallel(out, attnCtx, oW, hiddenSize, qRows) {
						return nil, fmt.Errorf("encoder BF16-fallback O GEMV rejected layer=%d pos=%d", layer, pos)
					}
				} else if !simd.GemvRowsParallel(out, attnCtx, oW, hiddenSize, qRows) {
					return nil, fmt.Errorf("encoder O GEMV rejected layer=%d pos=%d", layer, pos)
				}
				copy(hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
			}
		}

		// post_attention_norm + residual add
		postAttnNorm, err := loadFloatVector(weights, lb.PostAttentionLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], postAttnNorm, 1e-6) {
				return nil, fmt.Errorf("encoder post-attention RMSNorm rejected layer=%d offset=%d", layer, off)
			}
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}
		tAttention += time.Since(t0Attention)

		// FFN: save residual, pre_norm, dense MLP
		t0Dense := time.Now()
		copy(residual, hidden)
		preFFN, err := loadFloatVector(weights, lb.PreFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], preFFN, 1e-6) {
				return nil, fmt.Errorf("encoder pre-FFN RMSNorm rejected layer=%d offset=%d", layer, off)
			}
		}

		// Dense MLP: use BF16 native when available
		var gateW, upW, downW []float32
		var gateRows, gateCols int
		gateBF16, _, err := weights.RawBF16Tensor(lb.MLPGateProj.Name)
		if err != nil {
			return nil, err
		}
		upBF16, _, err := weights.RawBF16Tensor(lb.MLPUpProj.Name)
		if err != nil {
			return nil, err
		}
		downBF16, _, err := weights.RawBF16Tensor(lb.MLPDownProj.Name)
		if err != nil {
			return nil, err
		}
		useBF16MLP := gateBF16 != nil && upBF16 != nil && downBF16 != nil
		if lb.MLPGateProj != nil && len(lb.MLPGateProj.Shape) >= 2 {
			gateRows = lb.MLPGateProj.Shape[0]
			gateCols = lb.MLPGateProj.Shape[1]
		}
		if !useBF16MLP && fp8 == nil {
			var err2 error
			gateW, gateRows, gateCols, err2 = loadFloatMatrix(weights, lb.MLPGateProj)
			if err2 != nil {
				return nil, err2
			}
			upW, _, _, err2 = loadFloatMatrix(weights, lb.MLPUpProj)
			if err2 != nil {
				return nil, err2
			}
			downW, _, _, err2 = loadFloatMatrix(weights, lb.MLPDownProj)
			if err2 != nil {
				return nil, err2
			}
		}
		intermediate := gateRows
		mlpResult := make([]float32, len(hidden))
		ggufPrefillGPUMlp := fp8 == nil && !useBF16MLP && gpu.SgemmReady()
		if ggufPrefillGPUMlp {
			var resident *GGUFGPUMLPWeights
			if d.ggufDenseLayerResident(layer) {
				resident, err = residentGGUFGPUMLPWeights(layer, lb, gateW, upW, downW, hiddenSize, intermediate)
				if err != nil {
					return nil, err
				}
			} else {
				tmp := tempDense()
				gateBuf, err := tmp.Upload("encoder_mlp_gate", gateW, intermediate, hiddenSize)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU MLP gate upload rejected layer=%d: %w", layer, err)
				}
				upBuf, err := tmp.Upload("encoder_mlp_up", upW, intermediate, hiddenSize)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU MLP up upload rejected layer=%d: %w", layer, err)
				}
				downBuf, err := tmp.Upload("encoder_mlp_down", downW, hiddenSize, intermediate)
				if err != nil {
					return nil, fmt.Errorf("encoder temporary GPU MLP down upload rejected layer=%d: %w", layer, err)
				}
				resident = &GGUFGPUMLPWeights{Gate: gateBuf, Up: upBuf, Down: downBuf, Hidden: hiddenSize, Intermediate: intermediate}
			}
			midLen, okMid := checked.MulInt(positions, intermediate)
			if !okMid {
				return nil, fmt.Errorf("encoder GPU MLP mid buffer overflow positions=%d intermediate=%d", positions, intermediate)
			}
			gateBatch := make([]float32, midLen)
			upBatch := make([]float32, midLen)
			if err := batchedGPUGemmTransposed(gateBatch, hidden, positions, intermediate, hiddenSize, resident.Gate); err != nil {
				return nil, fmt.Errorf("encoder GPU MLP gate rejected layer=%d: %w", layer, err)
			}
			if err := batchedGPUGemmTransposed(upBatch, hidden, positions, intermediate, hiddenSize, resident.Up); err != nil {
				return nil, fmt.Errorf("encoder GPU MLP up rejected layer=%d: %w", layer, err)
			}
			actBatch := make([]float32, midLen)
			for pos := 0; pos < positions; pos++ {
				if !simd.GELUExactMulTo(actBatch[pos*intermediate:(pos+1)*intermediate], gateBatch[pos*intermediate:(pos+1)*intermediate], upBatch[pos*intermediate:(pos+1)*intermediate]) {
					return nil, fmt.Errorf("encoder GPU MLP activation rejected layer=%d pos=%d", layer, pos)
				}
			}
			if err := batchedGPUGemmTransposed(mlpResult, actBatch, positions, hiddenSize, intermediate, resident.Down); err != nil {
				return nil, fmt.Errorf("encoder GPU MLP down rejected layer=%d: %w", layer, err)
			}
		} else {
			gate := make([]float32, intermediate)
			up := make([]float32, intermediate)
			act := make([]float32, intermediate)
			mlpOut := make([]float32, hiddenSize)
			for off := 0; off < len(hidden); off += hiddenSize {
				row := hidden[off : off+hiddenSize]
				if fp8 != nil && layer < len(fp8.Layers) {
					fl := &fp8.Layers[layer]
					rowIn := row
					if diffusionGemmaFP8DynamicActivationEnabled() {
						rowIn = quantizeDynamicTokenRow(qAct, row)
					}
					if err := gpu.GemvFP8E4M3(gate, rowIn, fl.Gate); err != nil {
						return nil, fmt.Errorf("encoder FP8 MLP gate rejected layer=%d offset=%d: %w", layer, off, err)
					}
					if err := gpu.GemvFP8E4M3(up, rowIn, fl.Up); err != nil {
						return nil, fmt.Errorf("encoder FP8 MLP up rejected layer=%d offset=%d: %w", layer, off, err)
					}
					if !simd.GELUExactMulTo(act, gate, up) {
						return nil, fmt.Errorf("encoder FP8 MLP activation rejected layer=%d offset=%d", layer, off)
					}
					actIn := act
					if diffusionGemmaFP8DynamicActivationEnabled() {
						actIn = quantizeDynamicTokenRow(qAct, act)
					}
					if err := gpu.GemvFP8E4M3(mlpOut, actIn, fl.Down); err != nil {
						return nil, fmt.Errorf("encoder FP8 MLP down rejected layer=%d offset=%d: %w", layer, off, err)
					}
				} else if useBF16MLP {
					if !bf16GemvNarrow(gate, row, gateBF16, gateRows, gateCols) || !bf16GemvNarrow(up, row, upBF16, gateRows, gateCols) {
						return nil, fmt.Errorf("encoder BF16 MLP gate/up GEMV rejected layer=%d offset=%d", layer, off)
					}
					if !simd.GELUExactMulTo(act, gate, up) {
						return nil, fmt.Errorf("encoder BF16 MLP activation rejected layer=%d offset=%d", layer, off)
					}
					if !bf16GemvNarrow(mlpOut, act, downBF16, hiddenSize, gateRows) {
						return nil, fmt.Errorf("encoder BF16 MLP down GEMV rejected layer=%d offset=%d", layer, off)
					}
				} else {
					if !simd.GemvRowsParallel(gate, row, gateW, intermediate, gateCols) || !simd.GemvRowsParallel(up, row, upW, intermediate, gateCols) {
						return nil, fmt.Errorf("encoder MLP gate/up GEMV rejected layer=%d offset=%d", layer, off)
					}
					if !simd.GELUExactMulTo(act, gate, up) {
						return nil, fmt.Errorf("encoder MLP activation rejected layer=%d offset=%d", layer, off)
					}
					if !simd.GemvRowsParallel(mlpOut, act, downW, hiddenSize, intermediate) {
						return nil, fmt.Errorf("encoder MLP down GEMV rejected layer=%d offset=%d", layer, off)
					}
				}
				copy(mlpResult[off:off+hiddenSize], mlpOut)
			}
		}

		// post_feedforward_layernorm_1
		postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(mlpResult); off += hiddenSize {
			if !simd.RMSNormTo(mlpResult[off:off+hiddenSize], postNorm1, 1e-6) {
				return nil, fmt.Errorf("encoder MLP post_norm_1 rejected layer=%d offset=%d", layer, off)
			}
		}
		tDense += time.Since(t0Dense)

		// MoE from residual
		t0MoE := time.Now()
		if cap(encoderMoeResult) < len(hidden) {
			encoderMoeResult = make([]float32, len(hidden))
		}
		moeResult := encoderMoeResult[:len(hidden)]
		for i := range moeResult {
			moeResult[i] = 0
		}
		if d.GGUFExpertIndex != nil {
			topK := buffers.TopKExperts
			if topK <= 0 {
				topK = 8
			}
			topKSlots, okTopK := checked.MulInt(positions, topK)
			if !okTopK {
				return nil, fmt.Errorf("DiffusionGemma encoder GGUF MoE scratch overflow positions=%d experts=%d topK=%d", positions, d.GGUFExpertIndex.NumExperts, topK)
			}
			if cap(encoderGGUFRouter) < d.GGUFExpertIndex.NumExperts {
				encoderGGUFRouter = make([]float32, d.GGUFExpertIndex.NumExperts)
			}
			if cap(encoderGGUFNormedRows) < len(residual) {
				encoderGGUFNormedRows = make([]float32, len(residual))
			}
			if cap(encoderGGUFTopKIDs) < topKSlots {
				encoderGGUFTopKIDs = make([]int, topKSlots)
			}
			if cap(encoderGGUFTopKVals) < topKSlots {
				encoderGGUFTopKVals = make([]float32, topKSlots)
			}
			moeScratch := ForwardScratch{
				Residual:        residual,
				MoeOut:          moeResult,
				Router:          encoderGGUFRouter[:d.GGUFExpertIndex.NumExperts],
				Experts:         encoderGGUFNormedRows[:len(residual)],
				TopKIDs:         encoderGGUFTopKIDs[:topKSlots],
				TopKVals:        encoderGGUFTopKVals[:topKSlots],
				TopKExperts:     topK,
				GGUFExpertIndex: d.GGUFExpertIndex,
			}
			routerOp := LayerOp{Layer: layer, Type: lt, Kind: OpRouter}
			if err := runRouterFromResidual(routerOp, weights, moeScratch); err != nil {
				return nil, err
			}
			usedGPUExperts := false
			var normedRows []float32
			if !shouldSkipDoomedGGUFGPUExpertAttempt(d.GGUFExpertIndex, layer) {
				gpuAttemptStart := time.Now()
				usedGPUExperts, normedRows, err = runGGUFGPUExpertsIndexed(LayerOp{Layer: layer, Type: lt, Kind: OpExperts}, weights, moeScratch, d.GGUFExpertIndex)
				ggufExpertDispatchCounters.gpuAttemptNS.Add(uint64(time.Since(gpuAttemptStart).Nanoseconds()))
				if err != nil {
					return nil, err
				}
			}
			if !usedGPUExperts {
				ggufExpertDispatchCounters.cpuFallback.Add(1)
				cpuFallbackStart := time.Now()
				if len(normedRows) > 0 {
					err = runGGUFCPUExpertsIndexedWithNormedRows(LayerOp{Layer: layer, Type: lt, Kind: OpExperts}, weights, moeScratch, d.GGUFExpertIndex, normedRows)
				} else {
					err = runGGUFCPUExpertsIndexed(LayerOp{Layer: layer, Type: lt, Kind: OpExperts}, weights, moeScratch, d.GGUFExpertIndex)
				}
				if err != nil {
					return nil, err
				}
				ggufExpertDispatchCounters.cpuFallbackNS.Add(uint64(time.Since(cpuFallbackStart).Nanoseconds()))
			}
		} else {
			scaleVec, err := loadFloatVector(weights, lb.RouterScale)
			if err != nil {
				return nil, err
			}
			projW, projRows, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
			if err != nil {
				return nil, err
			}
			if len(scaleVec) != hiddenSize || projCols != hiddenSize || projRows <= 0 {
				return nil, fmt.Errorf("DiffusionGemma encoder router shape mismatch scale=%d proj=[%d,%d] hidden=%d", len(scaleVec), projRows, projCols, hiddenSize)
			}
			numExperts := projRows
			scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
			normBuf := make([]float32, hiddenSize)
			scored := make([]float32, numExperts)
			topK := buffers.TopKExperts
			if topK <= 0 {
				topK = 8 // DiffusionGemma-26B-A4B published default / GGUF n_expert_used fallback
			}
			topIDs := make([]int, topK)
			topVals := make([]float32, topK)

			preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
			if err != nil {
				return nil, err
			}
			nExperts, moeIntermediate := 0, 0
			if lb.ExpertsGateUpProj != nil && lb.ExpertsDownProj != nil && len(lb.ExpertsGateUpProj.Shape) == 3 && len(lb.ExpertsDownProj.Shape) == 3 {
				nExperts = lb.ExpertsGateUpProj.Shape[0]
				gateUpDim := lb.ExpertsGateUpProj.Shape[1]
				if nExperts <= 0 || gateUpDim <= 0 || gateUpDim%2 != 0 || lb.ExpertsGateUpProj.Shape[2] != hiddenSize {
					return nil, fmt.Errorf("DiffusionGemma encoder expert gate_up shape %v incompatible with hidden=%d", lb.ExpertsGateUpProj.Shape, hiddenSize)
				}
				moeIntermediate = gateUpDim / 2
				if lb.ExpertsDownProj.Shape[0] != nExperts || lb.ExpertsDownProj.Shape[1] != hiddenSize || lb.ExpertsDownProj.Shape[2] != moeIntermediate {
					return nil, fmt.Errorf("DiffusionGemma encoder expert down shape %v want [%d,%d,%d]", lb.ExpertsDownProj.Shape, nExperts, hiddenSize, moeIntermediate)
				}
			} else if d.ExpertIndex != nil {
				nExperts = d.ExpertIndex.NumExperts
				moeIntermediate = d.ExpertIndex.Intermediate
				if d.ExpertIndex.HiddenSize != hiddenSize || nExperts <= 0 || moeIntermediate <= 0 {
					return nil, fmt.Errorf("DiffusionGemma encoder FP8 expert index shape hidden=%d/%d experts=%d intermediate=%d", d.ExpertIndex.HiddenSize, hiddenSize, nExperts, moeIntermediate)
				}
			} else {
				return nil, fmt.Errorf("DiffusionGemma encoder expert tensor bindings missing layer %d", layer)
			}
			if d.ExpertIndex != nil && (layer >= d.ExpertIndex.NumLayers || len(d.ExpertIndex.entries[layer]) < nExperts) {
				return nil, fmt.Errorf("DiffusionGemma encoder FP8 expert index missing layer %d experts", layer)
			}

			for pos := 0; pos < positions; pos++ {
				resRow := residual[pos*hiddenSize : (pos+1)*hiddenSize]

				// Router: norm(residual) * scale * scalar_root_size, project, softmax, topk
				copy(normBuf, resRow)
				if !simd.RMSNormNoScaleTo(normBuf, 1e-6) {
					return nil, fmt.Errorf("DiffusionGemma encoder router norm rejected")
				}
				for i := range normBuf {
					normBuf[i] *= scaleVec[i] * scalarRootSize
				}
				if !simd.GemvRows(scored, normBuf, projW, numExperts, projCols) {
					return nil, fmt.Errorf("DiffusionGemma encoder router GEMV rejected")
				}
				softmaxInPlace(scored)
				for i := range topIDs {
					topIDs[i] = -1
					topVals[i] = float32(math.Inf(-1))
				}
				for expertID, score := range scored {
					insertTopK(topIDs, topVals, expertID, score)
				}
				var topKSum float32
				for _, v := range topVals {
					if v > float32(math.Inf(-1)) {
						topKSum += v
					}
				}
				// llama.cpp build_moe_ffn clamps the selected-weight sum to the
				// smallest positive F16 value before normalizing.
				if topKSum < 6.103515625e-5 {
					topKSum = 6.103515625e-5
				}
				for i := range topVals {
					if topVals[i] > float32(math.Inf(-1)) {
						topVals[i] /= topKSum
					}
				}
				if lb.RouterPerExpertScale != nil && d.GGUFExpertIndex == nil {
					// Safetensors router.per_expert_scale is equivalent to llama.cpp's
					// per-expert down_exps_s multiplication. In GGUF mode that scale is
					// already applied by RunGGUFExpertMLP.
					perExpert, err2 := loadFloatVector(weights, lb.RouterPerExpertScale)
					if err2 != nil {
						return nil, err2
					}
					for i, id := range topIDs {
						if id >= 0 && id < len(perExpert) {
							topVals[i] *= perExpert[id]
						}
					}
				}

				// Expert MLP from pre_norm_2(residual)
				normedRow := make([]float32, hiddenSize)
				copy(normedRow, resRow)
				if !simd.RMSNormTo(normedRow, preNorm2, 1e-6) {
					return nil, fmt.Errorf("DiffusionGemma encoder expert pre_norm_2 rejected")
				}

				dst := moeResult[pos*hiddenSize : (pos+1)*hiddenSize]
				eGate := make([]float32, moeIntermediate)
				eUp := make([]float32, moeIntermediate)
				eAct := make([]float32, moeIntermediate)
				eOut := make([]float32, hiddenSize)
				for k := 0; k < topK; k++ {
					expertID := topIDs[k]
					weight := topVals[k]
					if expertID < 0 || expertID >= nExperts {
						continue
					}
					// Use GPU experts for prewarmed pinned-prefix layers. For overflow
					// layers, prefer CPU indexed FP8/GGUF fallback instead of evicting the
					// resident prefix and thrashing before the next denoising step.
					gpuDone := false
					if expertCache != nil && fp8w != nil {
						useGPUExpert := true
						if expertCache.HasPinnedEntries() {
							useGPUExpert = expertCache.LayerFullyPinned(layer, nExperts)
						}
						var gateL, upL, downL *gpu.GPUFP8E4M3Linear
						if useGPUExpert {
							gateL, upL, downL = expertCache.Get(layer, expertID)
							if gateL == nil && !expertCache.HasPinnedEntries() {
								var cacheErr error
								gateL, upL, downL, cacheErr = expertCache.Put(layer, expertID, fp8w)
								if cacheErr != nil {
									gateL, upL, downL = nil, nil, nil
								}
							}
							gpuDone = gateL != nil && upL != nil && downL != nil
						}
						if gpuDone {
							expertIn := normedRow
							if diffusionGemmaFP8DynamicActivationEnabled() {
								expertIn = quantizeDynamicTokenRow(qAct, normedRow)
							}
							if err := gpu.GemvFP8E4M3(eGate, expertIn, gateL); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d FP8 gate: %w", expertID, err)
							}
							if err := gpu.GemvFP8E4M3(eUp, expertIn, upL); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d FP8 up: %w", expertID, err)
							}
							if !simd.GELUExactMulTo(eAct, eGate, eUp) {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d activation rejected", expertID)
							}
							downIn := eAct
							if diffusionGemmaFP8DynamicActivationEnabled() {
								downIn = quantizeDynamicTokenRow(qAct, eAct)
							}
							if err := gpu.GemvFP8E4M3(eOut, downIn, downL); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d FP8 down: %w", expertID, err)
							}
						}
					}
					if !gpuDone {
						// Try GGUF expert path, then pre-indexed FP8 CPU path, then fused safetensor fallback.
						if d.GGUFExpertIndex != nil {
							if err := d.GGUFExpertIndex.RunGGUFExpertMLP(layer, expertID, normedRow, eOut); err != nil {
								return nil, err
							}
						} else if d.ExpertIndex != nil {
							ep := d.ExpertIndex.entries[layer][expertID]
							expertIn := normedRow
							if diffusionGemmaFP8DynamicActivationEnabled() {
								expertIn = quantizeDynamicTokenRow(qAct, normedRow)
							}
							if err := ep.gate.GemvTo(expertIn, eGate); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d indexed FP8 gate: %w", expertID, err)
							}
							if err := ep.up.GemvTo(expertIn, eUp); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d indexed FP8 up: %w", expertID, err)
							}
							if !simd.GELUExactMulTo(eAct, eGate, eUp) {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d activation rejected", expertID)
							}
							downIn := eAct
							if diffusionGemmaFP8DynamicActivationEnabled() {
								downIn = quantizeDynamicTokenRow(qAct, eAct)
							}
							if err := ep.down.GemvTo(downIn, eOut); err != nil {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d indexed FP8 down: %w", expertID, err)
							}
						} else {
							guSlice, guRows, _, err := loadExpertSlice(weights, lb.ExpertsGateUpProj, expertID)
							if err != nil {
								return nil, err
							}
							dSlice, _, _, err := loadExpertSlice(weights, lb.ExpertsDownProj, expertID)
							if err != nil {
								return nil, err
							}
							gW := guSlice[:guRows/2*hiddenSize]
							uW := guSlice[guRows/2*hiddenSize:]
							if !simd.GemvRows(eGate, normedRow, gW, moeIntermediate, hiddenSize) || !simd.GemvRows(eUp, normedRow, uW, moeIntermediate, hiddenSize) {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d GEMV rejected", expertID)
							}
							if !simd.GELUExactMulTo(eAct, eGate, eUp) {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d activation rejected", expertID)
							}
							if !simd.GemvRows(eOut, eAct, dSlice, hiddenSize, moeIntermediate) {
								return nil, fmt.Errorf("DiffusionGemma encoder expert %d down GEMV rejected", expertID)
							}
						}
					}
					for i := range dst {
						dst[i] += weight * eOut[i]
					}
				}
			}

			// post_feedforward_layernorm_2
			postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
			if err != nil {
				return nil, err
			}
			for off := 0; off < len(moeResult); off += hiddenSize {
				if !simd.RMSNormTo(moeResult[off:off+hiddenSize], postNorm2, 1e-6) {
					return nil, fmt.Errorf("DiffusionGemma encoder post_norm_2 rejected")
				}
			}

		}
		tMoE += time.Since(t0MoE)

		// Combine MLP + MoE, apply post_feedforward_layernorm, add residual
		t0Other := time.Now()
		postFFN, err := loadFloatVector(weights, lb.PostFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for i := range hidden {
			hidden[i] = mlpResult[i] + moeResult[i]
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], postFFN, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma encoder post_feedforward norm rejected")
			}
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}

		// Layer scalar — encoder uses enc_layer_output_scale if available, else layer_output_scale
		scalarBinding := lb.EncLayerScalar
		if scalarBinding == nil {
			scalarBinding = lb.LayerScalar
		}
		if scalarBinding != nil {
			t, err := weights.CachedFloatTensor(scalarBinding.Name)
			if err != nil {
				return nil, err
			}
			if len(t.Shape) == 1 && t.Shape[0] == 1 {
				s := t.Data[0]
				for i := range hidden {
					hidden[i] *= s
				}
			}
		}

		tOther += time.Since(t0Other)
		totalNorm += tNorm
		totalAttention += tAttention
		totalDense += tDense
		totalMoE += tMoE
		totalOther += tOther
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma encoder: completed layer=%d elapsed=%s norm=%s attn=%s dense=%s moe=%s other=%s\n", layer, time.Since(layerStart).Round(time.Millisecond), tNorm.Round(time.Millisecond), tAttention.Round(time.Millisecond), tDense.Round(time.Millisecond), tMoE.Round(time.Millisecond), tOther.Round(time.Millisecond))
		}

		if layerTempDense != nil {
			layerTempDense.Close()
			layerTempDense = nil
			activeTempDenseSession = nil
		}

		// Wait for next layer's prefetch before evicting this layer
		if prefetchDone != nil {
			prefetchWaitStart := time.Now()
			<-prefetchDone
			totalPrefetchWait += time.Since(prefetchWaitStart)
		}

		// Evict non-resident layer weights
		if layer >= d.ResidentLayerPrefix {
			weights.EvictLayer(layer)
		}
	}

	if d.Progress {
		tempStats := ggufTempDenseUploadSnapshot().Sub(encoderTempDenseStatsStart)
		totalElapsed := time.Since(encoderStarted)
		accounted := totalNorm + totalAttention + totalDense + totalMoE + totalOther + totalPrefetchWait
		unaccounted := totalElapsed - accounted
		if unaccounted < 0 {
			unaccounted = 0
		}
		fmt.Fprintf(os.Stderr, "DiffusionGemma encoder summary: total=%s norm=%s attn=%s dense=%s moe=%s other=%s prefetch_wait=%s unaccounted=%s temp_dense_calls=%d hits=%d misses=%d temp_dense_bytes=%.1fMiB temp_transpose=%.1fs temp_upload=%.1fs e_attn=%d/%d e_mlp=%d/%d\n",
			totalElapsed.Round(time.Millisecond), totalNorm.Round(time.Millisecond), totalAttention.Round(time.Millisecond), totalDense.Round(time.Millisecond), totalMoE.Round(time.Millisecond), totalOther.Round(time.Millisecond), totalPrefetchWait.Round(time.Millisecond), unaccounted.Round(time.Millisecond),
			tempStats.Calls, tempStats.CacheHits, tempStats.CacheMisses, float64(tempStats.Bytes)/(1024*1024), float64(tempStats.TransposeNS)/1e9, float64(tempStats.UploadNS)/1e9,
			tempStats.EncoderAttnHits, tempStats.EncoderAttnCalls, tempStats.EncoderMLPHits, tempStats.EncoderMLPCalls)
		if d.GGUFExpertIndex != nil {
			cpuStats := ggufCPUExpertTimingSnapshot().Sub(encoderCPUExpertStatsStart)
			if cpuStats.Calls > 0 {
				fmt.Fprintf(os.Stderr, "DiffusionGemma encoder gguf_cpu_experts: calls=%d positions=%d work_items=%d active_experts=%d q4_direct_rows=%d q4_dequant_rows=%d q8_direct_rows=%d q8_dequant_rows=%d q4_batches(d=%s dq=%s) q8_batches(d=%s dq=%s) norm=%.1fs collect=%.1fs schedule=%.1fs gate=%.1fs act=%.1fs down=%.1fs scatter=%.1fs post=%.1fs\n",
					cpuStats.Calls, cpuStats.Positions, cpuStats.WorkItems, cpuStats.ActiveExperts, cpuStats.Q4DirectRows, cpuStats.Q4DequantRows, cpuStats.Q8DirectRows, cpuStats.Q8DequantRows,
					ggufCPUExpertBatchBucketsString(cpuStats.Q4DirectBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q4DequantBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q8DirectBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q8DequantBatches),
					float64(cpuStats.NormNS)/1e9, float64(cpuStats.CollectNS)/1e9, float64(cpuStats.ScheduleNS)/1e9, float64(cpuStats.GateNS)/1e9, float64(cpuStats.ActNS)/1e9, float64(cpuStats.DownNS)/1e9, float64(cpuStats.ScatterNS)/1e9, float64(cpuStats.PostNS)/1e9)
			}
			expertStats := ggufExpertDispatchStatsSnapshot().Sub(encoderGGUFExpertStatsStart)
			if expertStats.Total() > 0 {
				cacheUsed, cacheLimit := activeExpertMatrixCacheUsageBytes()
				activeAvg, workAvg, missingAvg, missingMiB, missingMaxMiB := expertStats.ActiveSetSummary()
				fmt.Fprintf(os.Stderr, "DiffusionGemma encoder gguf_experts: fused=%d legacy_grouped=%d cpu_fallback=%d gpu_attempt=%.1fs cpu_fallback_time=%.1fs cache=%.1f/%.1fMiB active_sets=%d active(avg/max)=%.1f/%d work(avg/max)=%.1f/%d q4_missing(avg/max)=%.1f/%d q4_missing_bytes=%.1fMiB max=%.1fMiB exceeds=%d q4(ptr/cache/transient_ptr/transient_pack/budget)=%d/%d/%d/%d/%d q4_budget=%.1fMiB/%dexperts q8(ptr/cache/transient_ptr/transient_pack/budget)=%d/%d/%d/%d/%d q8_budget=%.1fMiB/%dexperts\n",
					expertStats.FusedUsed, expertStats.LegacyGroupedUsed, expertStats.CPUFallback, float64(expertStats.GPUAttemptNS)/1e9, float64(expertStats.CPUFallbackNS)/1e9, float64(cacheUsed)/(1024*1024), float64(cacheLimit)/(1024*1024),
					expertStats.ActiveSetCalls, activeAvg, expertStats.ActiveSetMaxExperts, workAvg, expertStats.ActiveSetMaxWorkItems, missingAvg, expertStats.Q4MissingMaxExperts, missingMiB, missingMaxMiB, expertStats.Q4MissingBudgetExceeds,
					expertStats.Q4PointerTable, expertStats.Q4PackedCache, expertStats.Q4TransientPointer, expertStats.Q4TransientPacked, expertStats.Q4BudgetFallback, float64(expertStats.Q4BudgetBytes)/(1024*1024), expertStats.Q4BudgetExperts,
					expertStats.Q8PointerTable, expertStats.Q8PackedCache, expertStats.Q8TransientPointer, expertStats.Q8TransientPacked, expertStats.Q8BudgetFallback, float64(expertStats.Q8BudgetBytes)/(1024*1024), expertStats.Q8BudgetExperts)
			}
		}
	}
	return kvLayers, nil
}

// bf16GemvNarrow runs GEMV with BF16 weights and F32 hidden by narrowing
// hidden→BF16 and using BF16DotAsm. Avoids F32 weight decode.
func bf16GemvNarrow(out []float32, hidden []float32, wBF16 []uint16, rows, cols int) bool {
	xBF16 := simd.BF16FromF32Slice(hidden[:cols])
	return simd.GemvRowsBF16BF16Parallel(out, xBF16, wBF16, rows, cols)
}

// prefetchLayerWeights pre-warms the F32 cache for a layer's major projections
// by triggering CachedFloatTensor in a background goroutine. Also issues
// madvise(WILLNEED) on the raw mmap regions for kernel readahead.
func prefetchLayerWeights(weights *TextWeights, fp TextForwardPlan, layer int) <-chan struct{} {
	done := make(chan struct{})
	if layer < 0 || layer >= len(fp.Layers) {
		close(done)
		return done
	}
	go func() {
		defer close(done)
		lb := fp.Layers[layer]
		// Issue madvise(WILLNEED) on raw mmap for projection tensors
		// so kernel pages them in ahead of time, but do NOT decode to F32
		// (BF16 path reads raw mmap directly, FP8 path uses GPU).
		for _, b := range []*TensorBinding{
			lb.QProj, lb.KProj, lb.VProj, lb.OProj,
			lb.MLPGateProj, lb.MLPUpProj, lb.MLPDownProj,
		} {
			if b != nil {
				weights.RawTensor(b.Name) // madvise only, no decode
			}
		}
		// Both paths: prefetch norms/scalars (small, needed by all)
		for _, b := range []*TensorBinding{
			lb.InputLayerNorm, lb.PostAttentionLayerNorm,
			lb.PreFFNLayerNorm, lb.QNorm, lb.KNorm,
		} {
			if b != nil {
				weights.CachedFloatTensor(b.Name)
			}
		}
	}()
	return done
}
