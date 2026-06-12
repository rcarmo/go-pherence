package diffusiongemma

import (
	"fmt"
	"math"
	"os"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// EncodePrompt runs prompt token IDs through the same layers as the decoder
// (weights are tied) and captures per-layer K,V projections as encoder KV
// cache. The encoder uses causal attention (unlike the decoder which is
// bidirectional) and does not use self-conditioning.
func (d CPUDispatcher) EncodePrompt(promptIDs []int, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) ([]EncoderKVLayer, error) {
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
		if err := weights.PreloadLayerRange(0, d.ResidentLayerPrefix); err != nil {
			return nil, err
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

	// Run each layer
	for layer := 0; layer < numLayers; layer++ {
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

		// attention: compute Q,K,V projections
		qW, qRows, _, err := loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return nil, err
		}
		kW, kRows, _, err := loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return nil, err
		}
		var vW []float32
		vRows := kRows
		if lb.VProj != nil {
			vW, vRows, _, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return nil, err
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
			simd.GemvRows(q, h, qW, qRows, hiddenSize)
			simd.GemvRows(k, h, kW, kRows, hiddenSize)
			if lb.VProj != nil {
				simd.GemvRows(v, h, vW, vRows, hiddenSize)
			} else {
				copy(v, k)
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
		oW, _, _, err := loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return nil, err
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
			simd.GemvRows(out, attnCtx, oW, hiddenSize, qRows)
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

		// Dense MLP
		gateW, gateRows, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
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
		intermediate := gateRows
		gate := make([]float32, intermediate)
		up := make([]float32, intermediate)
		act := make([]float32, intermediate)
		mlpOut := make([]float32, hiddenSize)
		mlpResult := make([]float32, len(hidden))
		for off := 0; off < len(hidden); off += hiddenSize {
			row := hidden[off : off+hiddenSize]
			simd.GemvRows(gate, row, gateW, intermediate, gateCols)
			simd.GemvRows(up, row, upW, intermediate, gateCols)
			simd.GELUTanhMulTo(act, gate, up)
			simd.GemvRows(mlpOut, act, downW, hiddenSize, intermediate)
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
		gateUpAll, nExperts, gateUpDim, gateUpHidden, err := loadFloat3D(weights, lb.ExpertsGateUpProj)
		if err != nil {
			return nil, err
		}
		downAll, _, downHidden, downIntermediate, err := loadFloat3D(weights, lb.ExpertsDownProj)
		if err != nil {
			return nil, err
		}
		moeIntermediate := gateUpDim / 2
		_ = nExperts
		_ = gateUpHidden
		_ = downHidden
		_ = downIntermediate

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
				guSlice := gateUpAll[expertID*gateUpDim*gateUpHidden : (expertID+1)*gateUpDim*gateUpHidden]
				gW := guSlice[:moeIntermediate*hiddenSize]
				uW := guSlice[moeIntermediate*hiddenSize:]
				simd.GemvRows(eGate, normedRow, gW, moeIntermediate, hiddenSize)
				simd.GemvRows(eUp, normedRow, uW, moeIntermediate, hiddenSize)
				simd.GELUTanhMulTo(eAct, eGate, eUp)
				dSlice := downAll[expertID*downHidden*downIntermediate : (expertID+1)*downHidden*downIntermediate]
				simd.GemvRows(eOut, eAct, dSlice, hiddenSize, moeIntermediate)
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
			fmt.Fprintf(os.Stderr, "DiffusionGemma encoder: completed layer=%d\n", layer)
		}

		// Evict non-resident layer weights
		if layer >= d.ResidentLayerPrefix {
			weights.EvictLayer(layer)
		}
	}

	return kvLayers, nil
}
