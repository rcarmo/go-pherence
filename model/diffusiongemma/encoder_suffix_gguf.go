package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

func diffusionGemmaGGUFGPUSuffixEncoderEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_SUFFIX_ENCODER")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// EncodePromptSuffixGGUF appends prompt suffix KV on the CPU/reference path.
//
// Keep this available for llama.cpp parity and incremental-KV fixture checks.
// Production generation should still prefer GPU prompt prefill/append when
// available; callers must opt into CPUDispatcher explicitly.
func (d CPUDispatcher) EncodePromptSuffixGGUF(suffixIDs []int, prefixKV []EncoderKVLayer, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) ([]EncoderKVLayer, error) {
	if len(suffixIDs) == 0 {
		out := make([]EncoderKVLayer, len(prefixKV))
		copy(out, prefixKV)
		return out, nil
	}
	if d.GGUFExpertIndex == nil {
		return nil, fmt.Errorf("DiffusionGemma suffix encoder requires GGUFExpertIndex")
	}
	if len(prefixKV) == 0 || prefixKV[0].SeqLen <= 0 {
		return d.EncodePrompt(suffixIDs, weights, ops, buffers)
	}
	if weights == nil {
		return nil, fmt.Errorf("DiffusionGemma suffix encoder missing weights")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, fmt.Errorf("DiffusionGemma suffix encoder missing embed_tokens")
	}
	hiddenSize := fp.Globals.EmbedTokens.Shape[1]
	positions := len(suffixIDs)
	prefixLen := prefixKV[0].SeqLen
	if len(prefixKV) != len(fp.Layers) {
		return nil, fmt.Errorf("DiffusionGemma suffix encoder prefix KV layers=%d want %d", len(prefixKV), len(fp.Layers))
	}
	for i := range prefixKV {
		if prefixKV[i].SeqLen != prefixLen {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder prefix layer %d seq=%d want %d", i, prefixKV[i].SeqLen, prefixLen)
		}
	}

	hiddenElems, ok := checked.MulInt(positions, hiddenSize)
	if !ok {
		return nil, fmt.Errorf("DiffusionGemma suffix encoder hidden size overflow positions=%d hidden=%d", positions, hiddenSize)
	}
	hidden := make([]float32, hiddenElems)
	for i, token := range suffixIDs {
		row, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, token)
		if err != nil {
			return nil, err
		}
		if len(shape) != 1 || shape[0] != hiddenSize {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder embed row shape %v want [%d]", shape, hiddenSize)
		}
		if err := decodeFloatRowTo(hidden[i*hiddenSize:(i+1)*hiddenSize], row, dtype); err != nil {
			return nil, err
		}
	}
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range hidden {
		hidden[i] *= embedScale
	}

	numLayers := len(fp.Layers)
	suffixKV := make([]EncoderKVLayer, numLayers)
	residual := make([]float32, len(hidden))
	slidingWindow := buffers.SlidingWindow
	if slidingWindow <= 0 {
		slidingWindow = 1024
	}

	useGPUSuffix := diffusionGemmaGGUFGPUSuffixEncoderEnabled()
	if useGPUSuffix && d.Progress {
		fmt.Fprintf(os.Stderr, "DiffusionGemma suffix encoder: experimental GPU projection/dense path enabled; default CPU path is exact\n")
	}
	for layer := 0; layer < numLayers; layer++ {
		layerStart := time.Now()
		lb := fp.Layers[layer]
		lt := ""
		for _, op := range ops.Layers {
			if op.Layer == layer {
				lt = op.Type
				break
			}
		}
		copy(residual, hidden)
		normW, err := loadFloatVector(weights, lb.InputLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], normW, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder input norm rejected layer %d", layer)
			}
		}

		qW, qRows, qCols, err := loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return nil, err
		}
		kW, kRows, kCols, err := loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return nil, err
		}
		vRows, vCols := kRows, hiddenSize
		var vW []float32
		if lb.VProj != nil {
			vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return nil, err
			}
		}
		oW, oRows, oCols, err := loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return nil, err
		}
		if qCols != hiddenSize || kCols != hiddenSize || vCols != hiddenSize || oRows != hiddenSize || oCols != qRows {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder attention shape mismatch layer=%d", layer)
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
		if headDim <= 0 || len(kNorm) != headDim || qRows%headDim != 0 || kRows%headDim != 0 || vRows != kRows {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder head shape mismatch layer=%d", layer)
		}
		heads := qRows / headDim
		kvHeads := kRows / headDim
		if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder GQA mismatch layer=%d", layer)
		}
		prefix := prefixKV[layer]
		rowDim := kvHeads * headDim
		if prefix.KVHeads != kvHeads || prefix.HeadDim != headDim || len(prefix.Keys) < prefixLen*rowDim || len(prefix.Values) < prefixLen*rowDim {
			return nil, fmt.Errorf("DiffusionGemma suffix encoder prefix KV shape mismatch layer=%d", layer)
		}

		qAll := make([]float32, positions*qRows)
		kAll := make([]float32, positions*kRows)
		vAll := make([]float32, positions*vRows)
		if useGPUSuffix {
			qResult, err := batchedGPUGemm(qW, hidden, qRows, hiddenSize, positions)
			if err != nil {
				return nil, fmt.Errorf("DiffusionGemma suffix GPU Q GEMM layer=%d: %w", layer, err)
			}
			scatterGemmResult(qAll, qResult, qRows, positions)
			kResult, err := batchedGPUGemm(kW, hidden, kRows, hiddenSize, positions)
			if err != nil {
				return nil, fmt.Errorf("DiffusionGemma suffix GPU K GEMM layer=%d: %w", layer, err)
			}
			scatterGemmResult(kAll, kResult, kRows, positions)
			if lb.VProj != nil {
				vResult, err := batchedGPUGemm(vW, hidden, vRows, hiddenSize, positions)
				if err != nil {
					return nil, fmt.Errorf("DiffusionGemma suffix GPU V GEMM layer=%d: %w", layer, err)
				}
				scatterGemmResult(vAll, vResult, vRows, positions)
			}
		}
		ropeHalf := headDim / 2
		ropeTheta := 10000.0
		var ropeFactors []float32
		if lt == "full_attention" {
			ropeTheta = 1000000.0
			factors, err := fullAttentionRoPEFactors(weights, fp, headDim)
			if err != nil {
				return nil, err
			}
			ropeFactors = factors
		}
		ropeFreqs := simd.BuildRoPEFreqsWithFactors(prefixLen+positions, ropeHalf, headDim, ropeTheta, ropeFactors)
		for pos := 0; pos < positions; pos++ {
			h := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
			q := qAll[pos*qRows : (pos+1)*qRows]
			k := kAll[pos*kRows : (pos+1)*kRows]
			v := vAll[pos*vRows : (pos+1)*vRows]
			if !useGPUSuffix {
				if !simd.GemvRowsParallel(q, h, qW, qRows, hiddenSize) || !simd.GemvRowsParallel(k, h, kW, kRows, hiddenSize) {
					return nil, fmt.Errorf("DiffusionGemma suffix encoder Q/K GEMV rejected layer=%d pos=%d", layer, pos)
				}
				if lb.VProj != nil {
					if !simd.GemvRowsParallel(v, h, vW, vRows, hiddenSize) {
						return nil, fmt.Errorf("DiffusionGemma suffix encoder V GEMV rejected layer=%d pos=%d", layer, pos)
					}
				}
			}
			if lb.VProj == nil {
				copy(v, k)
			}
			for hh := 0; hh < heads; hh++ {
				if !simd.RMSNormTo(q[hh*headDim:(hh+1)*headDim], qNorm, 1e-6) {
					return nil, fmt.Errorf("DiffusionGemma suffix encoder Q norm rejected layer=%d", layer)
				}
			}
			for hh := 0; hh < kvHeads; hh++ {
				if !simd.RMSNormTo(k[hh*headDim:(hh+1)*headDim], kNorm, 1e-6) || !simd.RMSNormNoScaleTo(v[hh*headDim:(hh+1)*headDim], 1e-6) {
					return nil, fmt.Errorf("DiffusionGemma suffix encoder K/V norm rejected layer=%d", layer)
				}
			}
			absPos := prefixLen + pos
			if len(ropeFreqs) > 0 && ropeHalf > 0 {
				simd.ApplyRoPEPartial(q, ropeFreqs, absPos, heads, headDim, ropeHalf)
				simd.ApplyRoPEPartial(k, ropeFreqs, absPos, kvHeads, headDim, ropeHalf)
			}
		}
		suffixKV[layer] = EncoderKVLayer{Keys: append([]float32(nil), kAll...), Values: append([]float32(nil), vAll...), SeqLen: positions, KVHeads: kvHeads, HeadDim: headDim}

		group := heads / kvHeads
		attnCtx := make([]float32, qRows)
		out := make([]float32, hiddenSize)
		scores := make([]float32, prefixLen+positions)
		for pos := 0; pos < positions; pos++ {
			for i := range attnCtx {
				attnCtx[i] = 0
			}
			absPos := prefixLen + pos
			for hh := 0; hh < heads; hh++ {
				kvh := hh / group
				q := qAll[pos*qRows+hh*headDim : pos*qRows+(hh+1)*headDim]
				for j := 0; j < prefixLen+positions; j++ {
					if j > absPos || (lt == "sliding_attention" && absPos-j >= slidingWindow) {
						scores[j] = float32(math.Inf(-1))
						continue
					}
					if j < prefixLen {
						scores[j] = dot(q, prefix.Keys[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
					} else {
						sj := j - prefixLen
						if sj > pos {
							scores[j] = float32(math.Inf(-1))
						} else {
							scores[j] = dot(q, kAll[sj*kRows+kvh*headDim:sj*kRows+(kvh+1)*headDim])
						}
					}
				}
				softmaxInPlace(scores)
				dst := attnCtx[hh*headDim : (hh+1)*headDim]
				for j, score := range scores {
					var vv []float32
					if j < prefixLen {
						vv = prefix.Values[j*vRows+kvh*headDim : j*vRows+(kvh+1)*headDim]
					} else {
						sj := j - prefixLen
						vv = vAll[sj*vRows+kvh*headDim : sj*vRows+(kvh+1)*headDim]
					}
					for dd := range dst {
						dst[dd] += score * vv[dd]
					}
				}
			}
			if !simd.GemvRowsParallel(out, attnCtx, oW, hiddenSize, qRows) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder O GEMV rejected layer=%d pos=%d", layer, pos)
			}
			copy(hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
		}
		postAttnNorm, err := loadFloatVector(weights, lb.PostAttentionLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], postAttnNorm, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder post-attention norm rejected layer=%d", layer)
			}
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}

		copy(residual, hidden)
		preFFN, err := loadFloatVector(weights, lb.PreFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], preFFN, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder pre-FFN norm rejected layer=%d", layer)
			}
		}
		gateW, intermediate, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return nil, err
		}
		upW, _, _, err := loadFloatMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return nil, err
		}
		downW, _, _, err := loadFloatMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return nil, err
		}
		mlpResult := make([]float32, len(hidden))
		if useGPUSuffix {
			gateResult, err := batchedGPUGemm(gateW, hidden, intermediate, gateCols, positions)
			if err != nil {
				return nil, fmt.Errorf("DiffusionGemma suffix GPU dense gate layer=%d: %w", layer, err)
			}
			upResult, err := batchedGPUGemm(upW, hidden, intermediate, gateCols, positions)
			if err != nil {
				return nil, fmt.Errorf("DiffusionGemma suffix GPU dense up layer=%d: %w", layer, err)
			}
			actBatch := make([]float32, positions*intermediate)
			gateRow := make([]float32, intermediate)
			upRow := make([]float32, intermediate)
			for pos := 0; pos < positions; pos++ {
				for m := 0; m < intermediate; m++ {
					gateRow[m] = gateResult[m*positions+pos]
					upRow[m] = upResult[m*positions+pos]
				}
				if !diffusionGemmaGELUMulTo(actBatch[pos*intermediate:(pos+1)*intermediate], gateRow, upRow) {
					return nil, fmt.Errorf("DiffusionGemma suffix GPU dense activation rejected layer=%d", layer)
				}
			}
			downResult, err := batchedGPUGemm(downW, actBatch, hiddenSize, intermediate, positions)
			if err != nil {
				return nil, fmt.Errorf("DiffusionGemma suffix GPU dense down layer=%d: %w", layer, err)
			}
			scatterGemmResult(mlpResult, downResult, hiddenSize, positions)
		} else {
			gate := make([]float32, intermediate)
			up := make([]float32, intermediate)
			act := make([]float32, intermediate)
			mlpOut := make([]float32, hiddenSize)
			for off := 0; off < len(hidden); off += hiddenSize {
				row := hidden[off : off+hiddenSize]
				if !simd.GemvRowsParallel(gate, row, gateW, intermediate, gateCols) || !simd.GemvRowsParallel(up, row, upW, intermediate, gateCols) {
					return nil, fmt.Errorf("DiffusionGemma suffix encoder dense gate/up rejected layer=%d", layer)
				}
				if !diffusionGemmaGELUMulTo(act, gate, up) || !simd.GemvRowsParallel(mlpOut, act, downW, hiddenSize, intermediate) {
					return nil, fmt.Errorf("DiffusionGemma suffix encoder dense down rejected layer=%d", layer)
				}
				copy(mlpResult[off:off+hiddenSize], mlpOut)
			}
		}
		postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(mlpResult); off += hiddenSize {
			if !simd.RMSNormTo(mlpResult[off:off+hiddenSize], postNorm1, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder dense post-norm rejected layer=%d", layer)
			}
		}

		moeResult := make([]float32, len(hidden))
		scaleVec, err := loadFloatVector(weights, lb.RouterScale)
		if err != nil {
			return nil, err
		}
		projW, numExperts, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
		if err != nil {
			return nil, err
		}
		preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
		if err != nil {
			return nil, err
		}
		topK := buffers.TopKExperts
		if topK <= 0 {
			topK = 8
		}
		normBuf := make([]float32, hiddenSize)
		scored := make([]float32, numExperts)
		topIDs := make([]int, topK)
		topVals := make([]float32, topK)
		scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
		for pos := 0; pos < positions; pos++ {
			resRow := residual[pos*hiddenSize : (pos+1)*hiddenSize]
			copy(normBuf, resRow)
			if !simd.RMSNormNoScaleTo(normBuf, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder router norm rejected layer=%d", layer)
			}
			for i := range normBuf {
				normBuf[i] *= scaleVec[i] * scalarRootSize
			}
			if !simd.GemvRows(scored, normBuf, projW, numExperts, projCols) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder router GEMV rejected layer=%d", layer)
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
			if topKSum < 6.103515625e-5 {
				topKSum = 6.103515625e-5
			}
			for i := range topVals {
				if topVals[i] > float32(math.Inf(-1)) {
					topVals[i] /= topKSum
				}
			}
			normedRow := make([]float32, hiddenSize)
			copy(normedRow, resRow)
			if !simd.RMSNormTo(normedRow, preNorm2, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder expert pre-norm rejected layer=%d", layer)
			}
			dst := moeResult[pos*hiddenSize : (pos+1)*hiddenSize]
			expertOut := make([]float32, hiddenSize)
			for k := 0; k < topK; k++ {
				expertID := topIDs[k]
				if expertID < 0 || expertID >= d.GGUFExpertIndex.NumExperts {
					continue
				}
				if err := d.GGUFExpertIndex.RunGGUFExpertMLP(layer, expertID, normedRow, expertOut); err != nil {
					return nil, err
				}
				for i := range dst {
					dst[i] += topVals[k] * expertOut[i]
				}
			}
		}
		postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
		if err != nil {
			return nil, err
		}
		for off := 0; off < len(moeResult); off += hiddenSize {
			if !simd.RMSNormTo(moeResult[off:off+hiddenSize], postNorm2, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder MoE post-norm rejected layer=%d", layer)
			}
		}
		postFFN, err := loadFloatVector(weights, lb.PostFFNLayerNorm)
		if err != nil {
			return nil, err
		}
		for i := range hidden {
			hidden[i] = mlpResult[i] + moeResult[i]
		}
		for off := 0; off < len(hidden); off += hiddenSize {
			if !simd.RMSNormTo(hidden[off:off+hiddenSize], postFFN, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma suffix encoder post-FFN norm rejected layer=%d", layer)
			}
		}
		for i := range hidden {
			hidden[i] += residual[i]
		}
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
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma suffix encoder: completed layer=%d suffix=%d elapsed=%s\n", layer, positions, time.Since(layerStart).Round(time.Millisecond))
		}
	}
	return appendEncoderKVLayers(prefixKV, suffixKV)
}
