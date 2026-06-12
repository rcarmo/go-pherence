package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

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

		// attention: compute Q,K,V projections. Use binding shapes first so
		// the K3 A100 path can avoid eager F32 decode of FP8 weights.
		if lb.QProj == nil || lb.KProj == nil || len(lb.QProj.Shape) != 2 || len(lb.KProj.Shape) != 2 {
			return nil, fmt.Errorf("DiffusionGemma encoder missing Q/K projection bindings layer %d", layer)
		}
		qRows, qCols := lb.QProj.Shape[0], lb.QProj.Shape[1]
		kRows, kCols := lb.KProj.Shape[0], lb.KProj.Shape[1]
		if qCols != hiddenSize || kCols != hiddenSize {
			return nil, fmt.Errorf("DiffusionGemma encoder Q/K projection shape mismatch q=%v k=%v hidden=%d", lb.QProj.Shape, lb.KProj.Shape, hiddenSize)
		}
		var qW, kW, vW []float32
		vRows := kRows
		if lb.VProj != nil {
			if len(lb.VProj.Shape) != 2 || lb.VProj.Shape[1] != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma encoder V projection shape mismatch v=%v hidden=%d", lb.VProj.Shape, hiddenSize)
			}
			vRows = lb.VProj.Shape[0]
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
		qDone, kDone, vDone := false, false, false
		if lb.VProj != nil {
			if done, err := k3GemmManyRowsQ80([][]float32{qAll, kAll, vAll}, hidden, positions, weights, []*TensorBinding{lb.QProj, lb.KProj, lb.VProj}); err != nil {
				return nil, err
			} else if done {
				qDone, kDone, vDone = true, true, true
			}
		} else {
			if done, err := k3GemmManyRowsQ80([][]float32{qAll, kAll}, hidden, positions, weights, []*TensorBinding{lb.QProj, lb.KProj}); err != nil {
				return nil, err
			} else if done {
				qDone, kDone = true, true
			}
		}
		if !qDone {
			var err error
			qDone, err = k3GemmRowsQ80(qAll, hidden, positions, weights, lb.QProj)
			if err != nil {
				return nil, err
			}
		}
		if !qDone {
			var qCols int
			qW, qRows, qCols, err = loadFloatMatrix(weights, lb.QProj)
			if err != nil {
				return nil, err
			}
			if qCols != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma encoder Q projection cols=%d hidden=%d", qCols, hiddenSize)
			}
		}
		if !kDone {
			var err error
			kDone, err = k3GemmRowsQ80(kAll, hidden, positions, weights, lb.KProj)
			if err != nil {
				return nil, err
			}
		}
		if !kDone {
			var kCols int
			kW, kRows, kCols, err = loadFloatMatrix(weights, lb.KProj)
			if err != nil {
				return nil, err
			}
			if kCols != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma encoder K projection cols=%d hidden=%d", kCols, hiddenSize)
			}
		}
		if lb.VProj != nil && !vDone {
			var err error
			vDone, err = k3GemmRowsQ80(vAll, hidden, positions, weights, lb.VProj)
			if err != nil {
				return nil, err
			}
		}
		if lb.VProj != nil {
			if !vDone {
				var vCols int
				vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
				if err != nil {
					return nil, err
				}
				if vCols != hiddenSize {
					return nil, fmt.Errorf("DiffusionGemma encoder V projection cols=%d hidden=%d", vCols, hiddenSize)
				}
			}
		}

		for pos := 0; pos < positions; pos++ {
			h := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
			q := qAll[pos*qRows : (pos+1)*qRows]
			k := kAll[pos*kRows : (pos+1)*kRows]
			v := vAll[pos*vRows : (pos+1)*vRows]
			if !qDone {
				simd.GemvRows(q, h, qW, qRows, hiddenSize)
			}
			if !kDone {
				simd.GemvRows(k, h, kW, kRows, hiddenSize)
			}
			if lb.VProj != nil {
				if !vDone {
					simd.GemvRows(v, h, vW, vRows, hiddenSize)
				}
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
		var oW []float32
		attnAll := make([]float32, positions*qRows)
		runEncoderAttentionContextK3(attnAll, qAll, kAll, vAll, positions, heads, kvHeads, headDim, qRows, kRows, vRows, group)
		if done, err := k3GemmRowsQ80(hidden, attnAll, positions, weights, lb.OProj); err != nil {
			return nil, err
		} else if !done {
			var oRows, oCols int
			oW, oRows, oCols, err = loadFloatMatrix(weights, lb.OProj)
			if err != nil {
				return nil, err
			}
			if oRows != hiddenSize || oCols != qRows {
				return nil, fmt.Errorf("DiffusionGemma encoder O projection shape mismatch o=[%d,%d] hidden=%d q=%d", oRows, oCols, hiddenSize, qRows)
			}
			out := make([]float32, hiddenSize)
			for pos := 0; pos < positions; pos++ {
				attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
				simd.GemvRows(out, attnCtx, oW, hiddenSize, qRows)
				copy(hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
			}
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
		if lb.MLPGateProj == nil || lb.MLPUpProj == nil || lb.MLPDownProj == nil || len(lb.MLPGateProj.Shape) != 2 || len(lb.MLPUpProj.Shape) != 2 || len(lb.MLPDownProj.Shape) != 2 {
			return nil, fmt.Errorf("DiffusionGemma encoder missing MLP bindings layer %d", layer)
		}
		intermediate := lb.MLPGateProj.Shape[0]
		if lb.MLPGateProj.Shape[1] != hiddenSize || lb.MLPUpProj.Shape[0] != intermediate || lb.MLPUpProj.Shape[1] != hiddenSize || lb.MLPDownProj.Shape[0] != hiddenSize || lb.MLPDownProj.Shape[1] != intermediate {
			return nil, fmt.Errorf("DiffusionGemma encoder MLP shape mismatch gate=%v up=%v down=%v hidden=%d", lb.MLPGateProj.Shape, lb.MLPUpProj.Shape, lb.MLPDownProj.Shape, hiddenSize)
		}
		gateAll := make([]float32, positions*intermediate)
		upAll := make([]float32, positions*intermediate)
		if done, err := k3Gemm2RowsQ80(gateAll, upAll, hidden, positions, weights, lb.MLPGateProj, lb.MLPUpProj); err != nil {
			return nil, err
		} else if !done {
			gateW, gateRows, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
			if err != nil {
				return nil, err
			}
			upW, upRows, upCols, err := loadFloatMatrix(weights, lb.MLPUpProj)
			if err != nil {
				return nil, err
			}
			if gateRows != intermediate || gateCols != hiddenSize || upRows != intermediate || upCols != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma encoder MLP fallback shape mismatch")
			}
			for pos := 0; pos < positions; pos++ {
				row := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
				gate := gateAll[pos*intermediate : (pos+1)*intermediate]
				up := upAll[pos*intermediate : (pos+1)*intermediate]
				simd.GemvRows(gate, row, gateW, intermediate, hiddenSize)
				simd.GemvRows(up, row, upW, intermediate, hiddenSize)
			}
		}
		actAll := make([]float32, positions*intermediate)
		for pos := 0; pos < positions; pos++ {
			gate := gateAll[pos*intermediate : (pos+1)*intermediate]
			up := upAll[pos*intermediate : (pos+1)*intermediate]
			act := actAll[pos*intermediate : (pos+1)*intermediate]
			simd.GELUTanhMulTo(act, gate, up)
		}
		mlpResult := make([]float32, len(hidden))
		if done, err := k3GemmRowsQ80(mlpResult, actAll, positions, weights, lb.MLPDownProj); err != nil {
			return nil, err
		} else if !done {
			downW, downRows, downCols, err := loadFloatMatrix(weights, lb.MLPDownProj)
			if err != nil {
				return nil, err
			}
			if downRows != hiddenSize || downCols != intermediate {
				return nil, fmt.Errorf("DiffusionGemma encoder MLP down fallback shape mismatch")
			}
			mlpOut := make([]float32, hiddenSize)
			for pos := 0; pos < positions; pos++ {
				act := actAll[pos*intermediate : (pos+1)*intermediate]
				simd.GemvRows(mlpOut, act, downW, hiddenSize, intermediate)
				copy(mlpResult[pos*hiddenSize:(pos+1)*hiddenSize], mlpOut)
			}
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
		if lb.RouterProj == nil || len(lb.RouterProj.Shape) != 2 {
			return nil, fmt.Errorf("DiffusionGemma encoder missing router projection binding layer %d", layer)
		}
		numExperts, projCols := lb.RouterProj.Shape[0], lb.RouterProj.Shape[1]
		if numExperts <= 0 || projCols != hiddenSize {
			return nil, fmt.Errorf("DiffusionGemma encoder router shape mismatch proj=%v hidden=%d", lb.RouterProj.Shape, hiddenSize)
		}
		scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
		routerInput := make([]float32, positions*hiddenSize)
		for pos := 0; pos < positions; pos++ {
			in := routerInput[pos*hiddenSize : (pos+1)*hiddenSize]
			copy(in, residual[pos*hiddenSize:(pos+1)*hiddenSize])
			if !simd.RMSNormNoScaleTo(in, 1e-6) {
				return nil, fmt.Errorf("DiffusionGemma encoder router norm rejected")
			}
			for i := range in {
				in[i] *= scaleVec[i] * scalarRootSize
			}
		}
		scoredAll := make([]float32, positions*numExperts)
		if done, err := k3GemmRowsQ80(scoredAll, routerInput, positions, weights, lb.RouterProj); err != nil {
			return nil, err
		} else if !done {
			projW, projRows, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
			if err != nil {
				return nil, err
			}
			if projRows != numExperts || projCols != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma encoder router fallback shape mismatch proj=[%d,%d] expected=[%d,%d]", projRows, projCols, numExperts, hiddenSize)
			}
			for pos := 0; pos < positions; pos++ {
				simd.GemvRows(scoredAll[pos*numExperts:(pos+1)*numExperts], routerInput[pos*hiddenSize:(pos+1)*hiddenSize], projW, numExperts, hiddenSize)
			}
		}
		topK := 8 // from config

		preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
		if err != nil {
			return nil, err
		}
		layout, err := expertLayoutForLayer(weights, lb, hiddenSize)
		if err != nil {
			return nil, fmt.Errorf("DiffusionGemma encoder expert layout layer %d: %w", layer, err)
		}
		moeIntermediate := layout.intermediate
		topIDsAll := make([]int, positions*topK)
		topValsAll := make([]float32, positions*topK)

		for pos := 0; pos < positions; pos++ {
			// Router: batched norm(residual) * scale * scalar_root_size projection, then softmax/top-k.
			scored := scoredAll[pos*numExperts : (pos+1)*numExperts]
			softmaxInPlace(scored)
			ids := topIDsAll[pos*topK : (pos+1)*topK]
			vals := topValsAll[pos*topK : (pos+1)*topK]
			for i := range ids {
				ids[i] = -1
				vals[i] = float32(math.Inf(-1))
			}
			for expertID, score := range scored {
				insertTopK(ids, vals, expertID, score)
			}
			var topKSum float32
			for _, v := range vals {
				if v > float32(math.Inf(-1)) {
					topKSum += v
				}
			}
			if topKSum > 0 {
				for i := range vals {
					vals[i] /= topKSum
				}
			}
			if lb.RouterPerExpertScale != nil {
				perExpert, err2 := loadFloatVector(weights, lb.RouterPerExpertScale)
				if err2 != nil {
					return nil, err2
				}
				for i, id := range ids {
					if id >= 0 && id < len(perExpert) {
						vals[i] *= perExpert[id]
					}
				}
			}
		}

		expertsDone, err := k3RunPerExpertRowsA100(weights, layout, residual, topIDsAll, topValsAll, moeResult, preNorm2, hiddenSize, positions, topK)
		if err != nil {
			return nil, err
		}
		if !expertsDone {
			decodedExperts := map[int]decodedExpertWeights{}
			for pos := 0; pos < positions; pos++ {
				resRow := residual[pos*hiddenSize : (pos+1)*hiddenSize]

				// Expert MLP from pre_norm_2(residual)
				normedRow := make([]float32, hiddenSize)
				copy(normedRow, resRow)
				simd.RMSNormTo(normedRow, preNorm2, 1e-6)

				dst := moeResult[pos*hiddenSize : (pos+1)*hiddenSize]
				eGate := make([]float32, moeIntermediate)
				eUp := make([]float32, moeIntermediate)
				eAct := make([]float32, moeIntermediate)
				eOut := make([]float32, hiddenSize)
				ids := topIDsAll[pos*topK : (pos+1)*topK]
				vals := topValsAll[pos*topK : (pos+1)*topK]
				for k := 0; k < topK; k++ {
					expertID := ids[k]
					weight := vals[k]
					if expertID < 0 || expertID >= layout.nExperts {
						continue
					}
					ew, ok := decodedExperts[expertID]
					if !ok {
						ew, err = loadLayerExpertWeights(weights, lb, layout, expertID, hiddenSize)
						if err != nil {
							return nil, err
						}
						decodedExperts[expertID] = ew
					}
					simd.GemvRows(eGate, normedRow, ew.gateW, moeIntermediate, hiddenSize)
					simd.GemvRows(eUp, normedRow, ew.upW, moeIntermediate, hiddenSize)
					simd.GELUTanhMulTo(eAct, eGate, eUp)
					simd.GemvRows(eOut, eAct, ew.downW, hiddenSize, moeIntermediate)
					k3SaxpyV(weight, eOut, dst)
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

		// Evict non-resident layer weights
		if layer >= d.ResidentLayerPrefix {
			weights.EvictLayer(layer)
		}
	}

	return kvLayers, nil
}

func runEncoderAttentionContextK3(attnAll, qAll, kAll, vAll []float32, positions, heads, kvHeads, headDim, qRows, kRows, vRows, group int) {
	if positions <= 0 || heads <= 0 || headDim <= 0 {
		return
	}
	work := func(start, end int) {
		scores := make([]float32, positions)
		for pos := start; pos < end; pos++ {
			attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
			for i := range attnCtx {
				attnCtx[i] = 0
			}
			for hh := 0; hh < heads; hh++ {
				kvh := hh / group
				q := qAll[pos*qRows+hh*headDim : pos*qRows+(hh+1)*headDim]
				for j := 0; j < positions; j++ {
					if j > pos {
						scores[j] = float32(math.Inf(-1))
					} else {
						scores[j] = k3Dot(q, kAll[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
					}
				}
				k3SoftmaxInPlace(scores[:positions])
				dst := attnCtx[hh*headDim : (hh+1)*headDim]
				for j := 0; j < positions; j++ {
					vv := vAll[j*vRows+kvh*headDim : j*vRows+(kvh+1)*headDim]
					k3SaxpyV(scores[j], vv, dst)
				}
			}
		}
	}
	nw := 1
	if k3Enabled() && positions*heads >= 32 {
		nw = k3Threads()
		if nw > positions {
			nw = positions
		}
	}
	if nw <= 1 {
		work(0, positions)
		return
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		start := wid * positions / nw
		end := (wid + 1) * positions / nw
		go func() {
			defer wg.Done()
			work(start, end)
		}()
	}
	wg.Wait()
}
