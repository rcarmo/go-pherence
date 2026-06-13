package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// EncodePrompt runs prompt token IDs through the same layers as the decoder
// (weights are tied) and captures per-layer K,V projections as encoder KV
// cache. The encoder uses causal attention (unlike the decoder which is
// bidirectional) and does not use self-conditioning.
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
		if fp8 != nil {
			// GPU FP8 handles projections; only preload norms/scalars/router
			if err := weights.PreloadLayerRangeLightweight(0, d.ResidentLayerPrefix); err != nil {
				return nil, err
			}
		} else {
			if err := weights.PreloadLayerRange(0, d.ResidentLayerPrefix); err != nil {
				return nil, err
			}
		}
	}

	// Embed prompt tokens with scale
	hidden := make([]float32, positions*hiddenSize)
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

	// Run each layer with prefetch of next layer's weights
	var prefetchDone <-chan struct{}
	for layer := 0; layer < numLayers; layer++ {
		// Start prefetching layer+1 weights while this layer computes
		if layer+1 < numLayers {
			prefetchDone = prefetchLayerWeights(weights, fp, layer+1, fp8 != nil)
		}
		layerStart := time.Now()
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

		// attention: get shapes and try BF16 native path first
		qRows := lb.QProj.Shape[0]
		kRows := lb.KProj.Shape[0]
		vRows := kRows
		if lb.VProj != nil && len(lb.VProj.Shape) >= 1 {
			vRows = lb.VProj.Shape[0]
		}
		// Try BF16 zero-copy from mmap (avoids F32 decode entirely)
		qBF16, _, _ := weights.RawBF16Tensor(lb.QProj.Name)
		kBF16, _, _ := weights.RawBF16Tensor(lb.KProj.Name)
		useBF16 := qBF16 != nil && kBF16 != nil
		// Only decode F32 if BF16 native not available and not FP8
		var qW, kW, vW []float32
		if !useBF16 && fp8 == nil {
			var err2 error
			qW, _, _, err2 = loadFloatMatrix(weights, lb.QProj)
			if err2 != nil {
				return nil, err2
			}
			kW, _, _, err2 = loadFloatMatrix(weights, lb.KProj)
			if err2 != nil {
				return nil, err2
			}
			if lb.VProj != nil {
				vW, _, _, err2 = loadFloatMatrix(weights, lb.VProj)
				if err2 != nil {
					return nil, err2
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
		heads := qRows / headDim
		kvHeads := kRows / headDim

		qAll := make([]float32, positions*qRows)
		kAll := make([]float32, positions*kRows)
		vAll := make([]float32, positions*vRows)

		ropeHalf := headDim / 2
		ropeTheta := 10000.0
		if lt == "full_attention" {
			ropeHalf = headDim / 8
			ropeTheta = 1000000.0
		}
		ropeFreqs := simd.BuildRoPEFreqs(positions, ropeHalf, headDim, ropeTheta)

		for pos := 0; pos < positions; pos++ {
			h := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
			q := qAll[pos*qRows : (pos+1)*qRows]
			k := kAll[pos*kRows : (pos+1)*kRows]
			v := vAll[pos*vRows : (pos+1)*vRows]
			if fp8 != nil && layer < len(fp8.Layers) {
				fl := &fp8.Layers[layer]
				gpu.GemvFP8E4M3(q, h, fl.Q)
				gpu.GemvFP8E4M3(k, h, fl.K)
				if fl.V != nil {
					gpu.GemvFP8E4M3(v, h, fl.V)
				} else {
					copy(v, k)
				}
			} else if useBF16 {
				bf16GemvNarrow(q, h, qBF16, qRows, hiddenSize)
				bf16GemvNarrow(k, h, kBF16, kRows, hiddenSize)
				if lb.VProj != nil {
					vBF16, _, _ := weights.RawBF16Tensor(lb.VProj.Name)
					if vBF16 != nil {
						bf16GemvNarrow(v, h, vBF16, vRows, hiddenSize)
					} else {
						copy(v, k)
					}
				} else {
					copy(v, k)
				}
			} else {
				simd.GemvRowsParallel(q, h, qW, qRows, hiddenSize)
				simd.GemvRowsParallel(k, h, kW, kRows, hiddenSize)
				if lb.VProj != nil {
					simd.GemvRowsParallel(v, h, vW, vRows, hiddenSize)
				} else {
					copy(v, k)
				}
			}
			for hh := 0; hh < heads; hh++ {
				simd.RMSNormTo(q[hh*headDim:(hh+1)*headDim], qNorm, 1e-6)
			}
			for hh := 0; hh < kvHeads; hh++ {
				simd.RMSNormTo(k[hh*headDim:(hh+1)*headDim], kNorm, 1e-6)
				simd.RMSNormNoScaleTo(v[hh*headDim:(hh+1)*headDim], 1e-6)
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
		if !useBF16 && fp8 == nil {
			var err2 error
			oW, _, _, err2 = loadFloatMatrix(weights, lb.OProj)
			if err2 != nil {
				return nil, err2
			}
		}
		attnCtx := make([]float32, qRows)
		out := make([]float32, hiddenSize)
		scores := make([]float32, positions)
		for pos := 0; pos < positions; pos++ {
			for i := range attnCtx {
				attnCtx[i] = 0
			}
			for hh := 0; hh < heads; hh++ {
				kvh := hh / group
				q := qAll[pos*qRows+hh*headDim : pos*qRows+(hh+1)*headDim]
				for j := 0; j < positions; j++ {
					if j > pos {
						scores[j] = float32(math.Inf(-1)) // causal mask
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
			if fp8 != nil && layer < len(fp8.Layers) {
				gpu.GemvFP8E4M3(out, attnCtx, fp8.Layers[layer].O)
			} else if useBF16 {
				if oBF16, _, _ := weights.RawBF16Tensor(lb.OProj.Name); oBF16 != nil {
					bf16GemvNarrow(out, attnCtx, oBF16, hiddenSize, qRows)
				} else {
					simd.GemvRowsParallel(out, attnCtx, oW, hiddenSize, qRows)
				}
			} else {
				simd.GemvRowsParallel(out, attnCtx, oW, hiddenSize, qRows)
			}
			copy(hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
		}

		// post_attention_norm + residual add
		postAttnNorm, err := loadFloatVector(weights, lb.PostAttentionLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			simd.RMSNormTo(hidden[off:off+hiddenSize], postAttnNorm, 1e-6)
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}

		// FFN: save residual, pre_norm, dense MLP
		copy(residual, hidden)
		preFFN, err := loadFloatVector(weights, lb.PreFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			simd.RMSNormTo(hidden[off:off+hiddenSize], preFFN, 1e-6)
		}

		// Dense MLP: use BF16 native when available
		var gateW, upW, downW []float32
		var gateRows, gateCols int
		gateBF16, _, _ := weights.RawBF16Tensor(lb.MLPGateProj.Name)
		upBF16, _, _ := weights.RawBF16Tensor(lb.MLPUpProj.Name)
		downBF16, _, _ := weights.RawBF16Tensor(lb.MLPDownProj.Name)
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
		gate := make([]float32, intermediate)
		up := make([]float32, intermediate)
		act := make([]float32, intermediate)
		mlpOut := make([]float32, hiddenSize)
		mlpResult := make([]float32, len(hidden))
		for off := 0; off < len(hidden); off += hiddenSize {
			row := hidden[off : off+hiddenSize]
			if fp8 != nil && layer < len(fp8.Layers) {
				fl := &fp8.Layers[layer]
				gpu.GemvFP8E4M3(gate, row, fl.Gate)
				gpu.GemvFP8E4M3(up, row, fl.Up)
				simd.GELUTanhMulTo(act, gate, up)
				gpu.GemvFP8E4M3(mlpOut, act, fl.Down)
			} else if useBF16MLP {
				bf16GemvNarrow(gate, row, gateBF16, gateRows, gateCols)
				bf16GemvNarrow(up, row, upBF16, gateRows, gateCols)
				simd.GELUTanhMulTo(act, gate, up)
				bf16GemvNarrow(mlpOut, act, downBF16, hiddenSize, gateRows)
			} else {
				simd.GemvRowsParallel(gate, row, gateW, intermediate, gateCols)
				simd.GemvRowsParallel(up, row, upW, intermediate, gateCols)
				simd.GELUTanhMulTo(act, gate, up)
				simd.GemvRowsParallel(mlpOut, act, downW, hiddenSize, intermediate)
			}
			copy(mlpResult[off:off+hiddenSize], mlpOut)
		}

		// post_feedforward_layernorm_1
		postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(mlpResult); off += hiddenSize {
			simd.RMSNormTo(mlpResult[off:off+hiddenSize], postNorm1, 1e-6)
		}

		// MoE from residual
		moeResult := make([]float32, len(hidden))
		scaleVec, err := loadFloatVector(weights, lb.RouterScale)
		if err != nil {
			return nil, err
		}
		projW, projRows, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
		if err != nil {
			return nil, err
		}
		numExperts := projRows
		scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
		normBuf := make([]float32, hiddenSize)
		scored := make([]float32, numExperts)
		topK := 8 // from config
		topIDs := make([]int, topK)
		topVals := make([]float32, topK)

		preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
		if err != nil {
			return nil, err
		}
		if lb.ExpertsGateUpProj == nil || lb.ExpertsDownProj == nil || len(lb.ExpertsGateUpProj.Shape) != 3 || len(lb.ExpertsDownProj.Shape) != 3 {
			return nil, fmt.Errorf("DiffusionGemma encoder expert tensor bindings missing layer %d", layer)
		}
		nExperts := lb.ExpertsGateUpProj.Shape[0]
		gateUpDim := lb.ExpertsGateUpProj.Shape[1]
		moeIntermediate := gateUpDim / 2

		for pos := 0; pos < positions; pos++ {
			resRow := residual[pos*hiddenSize : (pos+1)*hiddenSize]

			// Router: norm(residual) * scale * scalar_root_size, project, softmax, topk
			copy(normBuf, resRow)
			simd.RMSNormNoScaleTo(normBuf, 1e-6)
			for i := range normBuf {
				normBuf[i] *= scaleVec[i] * scalarRootSize
			}
			simd.GemvRows(scored, normBuf, projW, numExperts, projCols)
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
			if topKSum > 0 {
				for i := range topVals {
					topVals[i] /= topKSum
				}
			}
			if lb.RouterPerExpertScale != nil {
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
			simd.RMSNormTo(normedRow, preNorm2, 1e-6)

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
				// Use GPU expert LRU for all encoder expert calls.
				// The cache thrashing between encoder and decoder is acceptable
				// because GPU expert upload is still faster than BF16 CPU decode.
				gpuDone := false
				if expertCache != nil && fp8w != nil {
					gateL, upL, downL := expertCache.Get(layer, expertID)
					if gateL == nil {
						var cacheErr error
						gateL, upL, downL, cacheErr = expertCache.Put(layer, expertID, fp8w)
						if cacheErr == nil {
							gpuDone = true
						}
					} else {
						gpuDone = true
					}
					if gpuDone {
						gpu.GemvFP8E4M3(eGate, normedRow, gateL)
						gpu.GemvFP8E4M3(eUp, normedRow, upL)
						simd.GELUTanhMulTo(eAct, eGate, eUp)
						gpu.GemvFP8E4M3(eOut, eAct, downL)
					}
				}
				if !gpuDone {
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
					simd.GemvRows(eGate, normedRow, gW, moeIntermediate, hiddenSize)
					simd.GemvRows(eUp, normedRow, uW, moeIntermediate, hiddenSize)
					simd.GELUTanhMulTo(eAct, eGate, eUp)
					simd.GemvRows(eOut, eAct, dSlice, hiddenSize, moeIntermediate)
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
			simd.RMSNormTo(moeResult[off:off+hiddenSize], postNorm2, 1e-6)
		}

		// Combine MLP + MoE, apply post_feedforward_layernorm, add residual
		postFFN, err := loadFloatVector(weights, lb.PostFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for i := range hidden {
			hidden[i] = mlpResult[i] + moeResult[i]
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			simd.RMSNormTo(hidden[off:off+hiddenSize], postFFN, 1e-6)
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}

		// Layer scalar
		if lb.LayerScalar != nil {
			t, err := weights.CachedFloatTensor(lb.LayerScalar.Name)
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

		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma encoder: completed layer=%d elapsed=%s\n", layer, time.Since(layerStart).Round(time.Millisecond))
		}

		// Wait for next layer's prefetch before evicting this layer
		if prefetchDone != nil {
			<-prefetchDone
		}

		// Evict non-resident layer weights
		if layer >= d.ResidentLayerPrefix {
			weights.EvictLayer(layer)
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
func prefetchLayerWeights(weights *TextWeights, fp TextForwardPlan, layer int, gpuFP8 bool) <-chan struct{} {
	done := make(chan struct{})
	if layer < 0 || layer >= len(fp.Layers) {
		close(done)
		return done
	}
	go func() {
		defer close(done)
		lb := fp.Layers[layer]
		if !gpuFP8 {
			// CPU path: prefetch and decode all projection tensors
			for _, b := range []*TensorBinding{
				lb.QProj, lb.KProj, lb.VProj, lb.OProj,
				lb.MLPGateProj, lb.MLPUpProj, lb.MLPDownProj,
			} {
				if b != nil {
					weights.RawTensor(b.Name)
				}
			}
			for _, b := range []*TensorBinding{
				lb.QProj, lb.KProj, lb.VProj, lb.OProj,
				lb.MLPGateProj, lb.MLPUpProj, lb.MLPDownProj,
			} {
				if b != nil {
					weights.CachedFloatTensor(b.Name)
				}
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
