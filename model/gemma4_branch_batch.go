package model

import (
	"context"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

// Gemma4BranchBatchResult contains the logits after the final forced token of
// each independent branch. Results retain input order.
type Gemma4BranchBatchResult struct {
	Logits [][]float32
}

// ScoreIndependentBranches executes forced-token sibling paths against this
// session's immutable prefilled trunk. Transformer projections are batched at
// each suffix depth. Each branch owns separate suffix K/V; attention gathers
// only the shared trunk plus that branch's own suffix, so siblings can never
// attend to each other. The session and its trunk K/V are not mutated.
func (s *Gemma4DecodeSession) ScoreIndependentBranches(ctx context.Context, branches [][]int) (Gemma4BranchBatchResult, error) {
	if err := s.readyToDecode(); err != nil {
		return Gemma4BranchBatchResult{}, err
	}
	if ctx == nil {
		return Gemma4BranchBatchResult{}, fmt.Errorf("nil context")
	}
	if s.state == nil || s.state.compressedKV != nil {
		return Gemma4BranchBatchResult{}, fmt.Errorf("Gemma4 independent branches require float KV")
	}
	if len(branches) == 0 {
		return Gemma4BranchBatchResult{}, fmt.Errorf("empty Gemma4 branch batch")
	}
	if len(branches) > 256 {
		return Gemma4BranchBatchResult{}, fmt.Errorf("Gemma4 branch batch=%d exceeds 256", len(branches))
	}
	maxDepth := 0
	for i, branch := range branches {
		if len(branch) == 0 {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d has no tokens", i)
		}
		if len(branch) > s.opts.MaxTokens {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d tokens=%d exceeds session max=%d", i, len(branch), s.opts.MaxTokens)
		}
		for j, token := range branch {
			if token < 0 || token >= s.model.Config.VocabSize {
				return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d token[%d]=%d outside vocab=%d", i, j, token, s.model.Config.VocabSize)
			}
		}
		if len(branch) > maxDepth {
			maxDepth = len(branch)
		}
	}
	if maxContext := s.model.Config.MaxSeqLen; maxContext > 0 && len(s.output)+maxDepth > maxContext {
		return Gemma4BranchBatchResult{}, fmt.Errorf("trunk %d + branch %d exceeds model context %d", len(s.output), maxDepth, maxContext)
	}
	return s.model.runGemma4IndependentBranchBatch(ctx, len(s.output), s.state.kvCacheK, s.state.kvCacheV, branches)
}

type gemma4BranchSuffixKV struct {
	k [][]float32
	v [][]float32
}

func (m *LlamaModel) runGemma4IndependentBranchBatch(ctx context.Context, trunkLen int, trunkK, trunkV [][]float32, branches [][]int) (Gemma4BranchBatchResult, error) {
	if err := m.validateGemma4IndependentBranchTrunk(trunkLen, trunkK, trunkV); err != nil {
		return Gemma4BranchBatchResult{}, err
	}
	if err := m.validateGemma4IndependentBranchLayers(); err != nil {
		return Gemma4BranchBatchResult{}, err
	}
	result := Gemma4BranchBatchResult{Logits: make([][]float32, len(branches))}
	suffix := make([]gemma4BranchSuffixKV, len(branches))
	for i := range suffix {
		suffix[i].k = make([][]float32, m.Config.NumLayers)
		suffix[i].v = make([][]float32, m.Config.NumLayers)
	}

	maxDepth := 0
	for _, branch := range branches {
		if len(branch) > maxDepth {
			maxDepth = len(branch)
		}
	}
	for depth := 0; depth < maxDepth; depth++ {
		if err := ctx.Err(); err != nil {
			return Gemma4BranchBatchResult{}, err
		}
		active := make([]int, 0, len(branches))
		for branch := range branches {
			if depth < len(branches[branch]) {
				active = append(active, branch)
			}
		}
		hiddenRows, err := m.runGemma4IndependentBranchDepth(ctx, trunkLen, depth, active, branches, trunkK, trunkV, suffix)
		if err != nil {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch depth %d: %w", depth, err)
		}
		var completedRows [][]float32
		var completedBranches []int
		for row, branch := range active {
			if depth+1 == len(branches[branch]) {
				completedRows = append(completedRows, hiddenRows[row])
				completedBranches = append(completedBranches, branch)
			}
		}
		if len(completedRows) == 0 {
			continue
		}
		_, logits, _, err := m.FinishCPUDecodeBatch(completedRows)
		if err != nil {
			return Gemma4BranchBatchResult{}, fmt.Errorf("finish completed branches: %w", err)
		}
		for i, branch := range completedBranches {
			result.Logits[branch] = logits[i]
		}
	}
	for i, logits := range result.Logits {
		if len(logits) != m.Config.VocabSize {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d logits=%d, want vocab=%d", i, len(logits), m.Config.VocabSize)
		}
	}
	return result, nil
}

func (m *LlamaModel) runGemma4IndependentBranchDepth(ctx context.Context, trunkLen, depth int, active []int, branches [][]int, trunkK, trunkV [][]float32, suffix []gemma4BranchSuffixKV) ([][]float32, error) {
	B, h := len(active), m.Config.HiddenSize
	hiddenFlat := make([]float32, B*h)
	tokens := make([]int, B)
	positions := make([]int, B)
	pli := make([][][]float32, B)
	for row, branch := range active {
		token := branches[branch][depth]
		tokens[row] = token
		positions[row] = trunkLen + depth
		hidden := hiddenFlat[row*h : (row+1)*h]
		if err := m.ScaledTokenEmbeddingInto(hidden, token); err != nil {
			return nil, fmt.Errorf("branch %d token embedding: %w", branch, err)
		}
		var err error
		pli[row], err = m.Gemma4PerLayerInputs(hidden, token)
		if err != nil {
			return nil, fmt.Errorf("branch %d per-layer inputs: %w", branch, err)
		}
	}
	return m.runGemma4IndependentBranchLayers(ctx, tokens, positions, active, hiddenFlat, pli, trunkLen, trunkK, trunkV, suffix)
}

func (m *LlamaModel) runGemma4IndependentBranchLayers(ctx context.Context, tokens, positions, active []int, hiddenFlat []float32, pli [][][]float32, trunkLen int, trunkK, trunkV [][]float32, suffix []gemma4BranchSuffixKV) ([][]float32, error) {
	B, h := len(active), m.Config.HiddenSize
	bHidden := append([]float32(nil), hiddenFlat...)
	bResidual := make([]float32, len(bHidden))
	maxQ, maxInter := 1, 1
	for l := 0; l < m.Config.NumLayers; l++ {
		headDim, err := m.LayerHeadDim(l)
		if err != nil {
			return nil, err
		}
		if q := m.Config.NumHeads * headDim; q > maxQ {
			maxQ = q
		}
		if inter := m.layerInterFor(&m.Layers[l]); inter > maxInter {
			maxInter = inter
		}
	}
	bAttnOut := make([]float32, B*maxQ)
	bOOut := make([]float32, B*h)
	bMlpIn := make([]float32, B*h)
	bGate := make([]float32, B*maxInter)
	bUp := make([]float32, B*maxInter)
	bDown := make([]float32, B*h)
	var bPLIGate, bPLIProj []float32
	if m.Config.HiddenPerLayer > 0 {
		bPLIGate = make([]float32, B*m.Config.HiddenPerLayer)
		bPLIProj = make([]float32, B*h)
	}
	eps := float32(m.Config.RMSNormEps)

	for l := 0; l < m.Config.NumLayers; l++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		layer := &m.Layers[l]
		copy(bResidual, bHidden)
		qkv, err := m.projectMTPVerifierLayerQKVRows(tokens, positions, l, bHidden)
		if err != nil {
			return nil, err
		}
		kvLayer := l
		if !qkv.HasKV {
			kvLayer = layer.KVSourceLayer
			sourceDim, err := m.LayerKVDim(kvLayer)
			if err != nil {
				return nil, err
			}
			if sourceDim != qkv.KVDim {
				return nil, fmt.Errorf("layer %d shared KV source %d dim=%d, want %d", l, kvLayer, sourceDim, qkv.KVDim)
			}
		}
		for row, branch := range active {
			if qkv.HasKV {
				suffix[branch].k[l] = append(suffix[branch].k[l], qkv.K[row*qkv.KVDim:(row+1)*qkv.KVDim]...)
				suffix[branch].v[l] = append(suffix[branch].v[l], qkv.V[row*qkv.KVDim:(row+1)*qkv.KVDim]...)
			}
			start := 0
			end := positions[row] + 1
			if m.Config.SlidingWindow > 0 && len(m.Config.LayerTypes) > l && m.Config.LayerTypes[l] == "sliding_attention" && end > m.Config.SlidingWindow {
				start = end - m.Config.SlidingWindow
			}
			kCache, vCache, err := gatherGemma4BranchAttention(trunkLen, start, end, qkv.KVDim, trunkK[kvLayer], trunkV[kvLayer], suffix[branch].k[kvLayer], suffix[branch].v[kvLayer])
			if err != nil {
				return nil, fmt.Errorf("layer %d branch %d attention: %w", l, branch, err)
			}
			attnLen := end - start
			out := bAttnOut[row*qkv.QDim : (row+1)*qkv.QDim]
			gqaAttentionScaleSoftcapInto(out, make([]float32, attnLen), qkv.Q[row*qkv.QDim:(row+1)*qkv.QDim], kCache, vCache, attnLen, m.Config.NumHeads, qkv.KVHeads, qkv.HeadDim, attentionScale(m.Config, qkv.HeadDim), attentionLogitSoftcap(m.Config))
		}
		if !m.projBatchAny(bOOut[:B*h], bAttnOut[:B*qkv.QDim], B, layer.OW, layer.OWm, layer.OWq, layer.OWGGUF, qkv.QDim, h) {
			return nil, fmt.Errorf("layer %d O projection rejected", l)
		}
		if layer.PreFFNNorm != nil {
			for row := 0; row < B; row++ {
				o := bOOut[row*h : (row+1)*h]
				rmsNormInPlace(o, layer.PostNorm.Data(), eps)
				simd.VecAdd(bHidden[row*h:(row+1)*h], bResidual[row*h:(row+1)*h], o)
			}
			copy(bResidual, bHidden)
			for row := 0; row < B; row++ {
				in := bMlpIn[row*h : (row+1)*h]
				copy(in, bHidden[row*h:(row+1)*h])
				rmsNormInPlace(in, layer.PreFFNNorm.Data(), eps)
			}
		} else {
			for row := 0; row < B; row++ {
				hid := bHidden[row*h : (row+1)*h]
				simd.VecAdd(hid, bResidual[row*h:(row+1)*h], bOOut[row*h:(row+1)*h])
			}
			copy(bResidual, bHidden)
			for row := 0; row < B; row++ {
				rmsNormInPlace(bHidden[row*h:(row+1)*h], layer.PostNorm.Data(), eps)
			}
			copy(bMlpIn, bHidden)
		}
		inter := m.layerInterFor(layer)
		if !m.projBatchAny(bGate[:B*inter], bMlpIn[:B*h], B, layer.GateW, layer.GateWm, layer.GateWq, layer.GateWGGUF, h, inter) || !m.projBatchAny(bUp[:B*inter], bMlpIn[:B*h], B, layer.UpW, layer.UpWm, layer.UpWq, layer.UpWGGUF, h, inter) {
			return nil, fmt.Errorf("layer %d MLP gate/up rejected", l)
		}
		for row := 0; row < B; row++ {
			gate := bGate[row*inter : (row+1)*inter]
			up := bUp[row*inter : (row+1)*inter]
			if m.Config.HiddenAct == "gelu_pytorch_tanh" {
				ggmlGELUMulInPlace(gate, up)
			} else {
				simd.VecSiLUMul(gate, gate, up)
			}
		}
		if !m.projBatchAny(bDown[:B*h], bGate[:B*inter], B, layer.DownW, layer.DownWm, layer.DownWq, layer.DownWGGUF, inter, h) {
			return nil, fmt.Errorf("layer %d MLP down rejected", l)
		}
		for row := 0; row < B; row++ {
			down := bDown[row*h : (row+1)*h]
			if layer.PostFFNNorm != nil {
				rmsNormInPlace(down, layer.PostFFNNorm.Data(), eps)
			}
			simd.VecAdd(bHidden[row*h:(row+1)*h], bResidual[row*h:(row+1)*h], down)
		}
		if (layer.PLIGate != nil || layer.PLIGateGGUF != nil) && pli != nil {
			hpl := m.Config.HiddenPerLayer
			if hpl <= 0 || len(layer.PLIPostNorm) < h {
				return nil, fmt.Errorf("layer %d invalid per-layer input dimensions", l)
			}
			if layer.PLIGateGGUF != nil {
				if !m.projBatchAny(bPLIGate[:B*hpl], bHidden[:B*h], B, nil, nil, nil, layer.PLIGateGGUF, h, hpl) {
					return nil, fmt.Errorf("layer %d GGUF PLI gate rejected", l)
				}
			} else if len(layer.PLIGate) < hpl*h || !simd.GemmRowsParallel(bPLIGate[:B*hpl], bHidden[:B*h], layer.PLIGate, B, hpl, h) {
				return nil, fmt.Errorf("layer %d PLI gate rejected", l)
			}
			for row := 0; row < B; row++ {
				if l >= len(pli[row]) || len(pli[row][l]) < hpl {
					return nil, fmt.Errorf("layer %d row %d missing PLI", l, row)
				}
				ggmlGELUMulInPlace(bPLIGate[row*hpl:(row+1)*hpl], pli[row][l][:hpl])
			}
			if layer.PLIProjGGUF != nil {
				if !m.projBatchAny(bPLIProj[:B*h], bPLIGate[:B*hpl], B, nil, nil, nil, layer.PLIProjGGUF, hpl, h) {
					return nil, fmt.Errorf("layer %d GGUF PLI projection rejected", l)
				}
			} else if len(layer.PLIProj) < h*hpl || !simd.GemmRowsParallel(bPLIProj[:B*h], bPLIGate[:B*hpl], layer.PLIProj, B, h, hpl) {
				return nil, fmt.Errorf("layer %d PLI projection rejected", l)
			}
			for row := 0; row < B; row++ {
				proj := bPLIProj[row*h : (row+1)*h]
				rmsNormInPlace(proj, layer.PLIPostNorm, eps)
				simd.VecAdd(bHidden[row*h:(row+1)*h], bHidden[row*h:(row+1)*h], proj)
			}
		}
		if layer.LayerScalar != 1 {
			for row := 0; row < B; row++ {
				simd.VecScale(bHidden[row*h:(row+1)*h], bHidden[row*h:(row+1)*h], layer.LayerScalar)
			}
		}
	}
	out := make([][]float32, B)
	for row := range out {
		out[row] = append([]float32(nil), bHidden[row*h:(row+1)*h]...)
	}
	return out, nil
}

func gatherGemma4BranchAttention(trunkLen, start, end, kvDim int, trunkK, trunkV, suffixK, suffixV []float32) ([]float32, []float32, error) {
	if trunkLen < 0 || start < 0 || end <= start || kvDim <= 0 || end > trunkLen+len(suffixK)/kvDim || len(suffixK) != len(suffixV) {
		return nil, nil, fmt.Errorf("invalid range [%d,%d) trunk=%d suffix=%d/%d kvDim=%d", start, end, trunkLen, len(suffixK), len(suffixV), kvDim)
	}
	if len(trunkK) != trunkLen*kvDim || len(trunkV) != trunkLen*kvDim {
		return nil, nil, fmt.Errorf("trunk K/V=%d/%d, want %d", len(trunkK), len(trunkV), trunkLen*kvDim)
	}
	rows := end - start
	k := make([]float32, rows*kvDim)
	v := make([]float32, rows*kvDim)
	for pos := start; pos < end; pos++ {
		dst := (pos - start) * kvDim
		if pos < trunkLen {
			copy(k[dst:dst+kvDim], trunkK[pos*kvDim:(pos+1)*kvDim])
			copy(v[dst:dst+kvDim], trunkV[pos*kvDim:(pos+1)*kvDim])
			continue
		}
		src := (pos - trunkLen) * kvDim
		copy(k[dst:dst+kvDim], suffixK[src:src+kvDim])
		copy(v[dst:dst+kvDim], suffixV[src:src+kvDim])
	}
	return k, v, nil
}

func (m *LlamaModel) validateGemma4IndependentBranchTrunk(trunkLen int, trunkK, trunkV [][]float32) error {
	if m == nil {
		return fmt.Errorf("nil independent branch model")
	}
	if m.Config.ModelType != "gemma4_text" {
		return fmt.Errorf("independent branch model type=%q, want gemma4_text", m.Config.ModelType)
	}
	if trunkLen <= 0 || len(trunkK) != m.Config.NumLayers || len(trunkV) != m.Config.NumLayers {
		return fmt.Errorf("invalid branch trunk len=%d K/V layers=%d/%d want=%d", trunkLen, len(trunkK), len(trunkV), m.Config.NumLayers)
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		dim, err := m.LayerKVDim(l)
		if err != nil {
			return err
		}
		want := trunkLen * dim
		if len(trunkK[l]) != want || len(trunkV[l]) != want {
			return fmt.Errorf("branch trunk layer %d K/V=%d/%d, want %d", l, len(trunkK[l]), len(trunkV[l]), want)
		}
	}
	return nil
}

func (m *LlamaModel) validateGemma4IndependentBranchLayers() error {
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		if layer.IsMoE || layer.InputNorm == nil || layer.PostNorm == nil {
			return fmt.Errorf("layer %d unsupported by independent branch batch", l)
		}
		if !hasMTPVerifierProjection(layer.QW, layer.QWm, layer.QWq, layer.QWGGUF) || !hasMTPVerifierProjection(layer.OW, layer.OWm, layer.OWq, layer.OWGGUF) || !hasMTPVerifierProjection(layer.GateW, layer.GateWm, layer.GateWq, layer.GateWGGUF) || !hasMTPVerifierProjection(layer.UpW, layer.UpWm, layer.UpWq, layer.UpWGGUF) || !hasMTPVerifierProjection(layer.DownW, layer.DownWm, layer.DownWq, layer.DownWGGUF) {
			return fmt.Errorf("layer %d missing independent branch projection", l)
		}
		if layer.HasKV && (!hasMTPVerifierProjection(layer.KW, layer.KWm, layer.KWq, layer.KWGGUF) || !(m.Config.AttentionKEqV && ((layer.KW != nil && (layer.VW == nil || layer.VW == layer.KW)) || (layer.KWm != nil && (layer.VWm == nil || layer.VWm == layer.KWm)) || (layer.KWq != nil && (layer.VWq == nil || layer.VWq == layer.KWq)) || (layer.KWGGUF != nil && (layer.VWGGUF == nil || layer.VWGGUF == layer.KWGGUF)))) && !hasMTPVerifierProjection(layer.VW, layer.VWm, layer.VWq, layer.VWGGUF)) {
			return fmt.Errorf("layer %d missing independent branch K/V projection", l)
		}
		if !layer.HasKV {
			if layer.KVSourceLayer < 0 || layer.KVSourceLayer >= l || !m.Layers[layer.KVSourceLayer].HasKV {
				return fmt.Errorf("layer %d invalid shared KV source %d", l, layer.KVSourceLayer)
			}
			sourceDim, err := m.LayerKVDim(layer.KVSourceLayer)
			if err != nil {
				return err
			}
			headDim, err := m.LayerHeadDim(l)
			if err != nil {
				return err
			}
			if sourceDim != gemmacfg.LayerKVHeads(m.Config, l)*headDim {
				return fmt.Errorf("layer %d shared KV source %d dim=%d does not match attention", l, layer.KVSourceLayer, sourceDim)
			}
		}
	}
	return nil
}
