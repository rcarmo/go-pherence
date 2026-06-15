package model

import (
	"fmt"
	"math"
	"os"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// RunMTPVerifierBatchForward is the verifier-batch execution entry point. It
// consumes materialized MTPVerifierBatchInputs for all verifier rows. The current
// lowering executes rows/layers sequentially while preserving the explicit batch
// and attention-plan contract; future SIMD/GPU work can replace the inner layer
// loop without changing callers.
func (m *LlamaModel) RunMTPVerifierBatchForward(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierBatchForwardInputs(batch, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
	finalHiddenRows, ok, err := m.runMTPVerifierBatchLayers(batch, kvCacheK, kvCacheV)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if !ok {
		finalHiddenRows, err = m.runMTPVerifierBatchRowsSequential(batch, kvCacheK, kvCacheV)
		if err != nil {
			return MTPVerifierResult{}, err
		}
	}
	finalActivations, logitsRows, _, err := m.FinishCPUDecodeBatch(finalHiddenRows)
	if err != nil {
		return MTPVerifierResult{}, fmt.Errorf("verifier batch decode finish: %w", err)
	}
	finalActivation := finalActivations[len(finalActivations)-1]
	return NewMTPVerifierResultForModel(m, batch.Plan.InputToken, batch.Plan.DraftedTokens, logitsRows, finalActivation)
}

func (m *LlamaModel) runMTPVerifierBatchRowsSequential(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) ([][]float32, error) {
	finalHiddenRows := make([][]float32, len(batch.HiddenRows))
	attnRows := batch.Scratch.MaxAttentionRows
	if attnRows < 1 {
		attnRows = 1
	}
	attnOutWidth := batch.Scratch.MaxQDim
	if attnOutWidth < 1 {
		attnOutWidth = m.Config.NumHeads * m.Config.HeadDim
	}
	attnScoresScratch := make([]float32, attnRows)
	attnOutScratch := make([]float32, attnOutWidth)
	for i, row := range batch.HiddenRows {
		hidden := append([]float32(nil), row...)
		pos := batch.Plan.Positions[i]
		perLayerInputs := batch.PerLayerInputs[i]
		for l := 0; l < m.Config.NumLayers; l++ {
			if perLayerInputs != nil {
				var err error
				hidden, err = m.forwardMTPPromptLayer(hidden, perLayerInputs, l, pos, kvCacheK, kvCacheV, attnScoresScratch, attnOutScratch)
				if err != nil {
					return nil, fmt.Errorf("verifier batch forward layer %d at position %d: %w", l, pos, err)
				}
				continue
			}
			hidden = m.ForwardLayer(hidden, l, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return nil, fmt.Errorf("verifier batch forward layer %d at position %d failed", l, pos)
			}
		}
		finalHiddenRows[i] = hidden
	}
	return finalHiddenRows, nil
}

func (m *LlamaModel) runMTPVerifierBatchLayers(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) ([][]float32, bool, error) {
	if !m.mtpVerifierBatchLayerEligible(batch) {
		return nil, false, nil
	}
	B, h := batch.Scratch.Batch, batch.Scratch.HiddenSize
	bHidden := append([]float32(nil), batch.HiddenFlat...)
	bResidual := make([]float32, len(bHidden))
	bAttnOut := make([]float32, B*batch.Scratch.MaxQDim)
	bOOut := make([]float32, B*h)
	bMlpIn := make([]float32, B*h)
	bGate := make([]float32, B*batch.Scratch.MaxIntermediate)
	bUp := make([]float32, B*batch.Scratch.MaxIntermediate)
	bDown := make([]float32, B*h)
	attnScores := make([]float32, batch.Scratch.MaxAttentionRows)
	isGemma := m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text"
	eps := float32(m.Config.RMSNormEps)
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		copy(bResidual, bHidden)
		qkv, err := m.ProjectMTPVerifierLayerQKVBatch(batch, l, bHidden)
		if err != nil {
			return nil, true, err
		}
		if qkv.HasKV {
			for b := 0; b < B; b++ {
				kvCacheK[l] = append(kvCacheK[l], qkv.K[b*qkv.KVDim:(b+1)*qkv.KVDim]...)
				kvCacheV[l] = append(kvCacheV[l], qkv.V[b*qkv.KVDim:(b+1)*qkv.KVDim]...)
			}
		}
		lp := batch.Attention.Layers[l]
		for b := 0; b < B; b++ {
			start, end := lp.KVStart[b], lp.KVEndExclusive[b]
			attnSeqLen := end - start
			if attnSeqLen <= 0 {
				return nil, true, fmt.Errorf("verifier batch layer %d row %d invalid attention range [%d,%d)", l, b, start, end)
			}
			kOff, kEnd := start*qkv.KVDim, end*qkv.KVDim
			if kEnd > len(kvCacheK[l]) || kEnd > len(kvCacheV[l]) {
				return nil, true, fmt.Errorf("verifier batch layer %d row %d KV range [%d,%d) exceeds K/V=%d/%d", l, b, kOff, kEnd, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			scale := float32(1.0 / math.Sqrt(float64(qkv.HeadDim)))
			if m.Config.ModelType == "gemma4_text" {
				scale = 1.0
			}
			gqaAttentionScaleInto(bAttnOut[b*qkv.QDim:(b+1)*qkv.QDim], attnScores[:attnSeqLen], qkv.Q[b*qkv.QDim:(b+1)*qkv.QDim], kvCacheK[l][kOff:kEnd], kvCacheV[l][kOff:kEnd], attnSeqLen, m.Config.NumHeads, qkv.KVHeads, qkv.HeadDim, scale)
		}
		if !m.projBatch(bOOut[:B*h], bAttnOut[:B*qkv.QDim], B, layer.OW, layer.OWm, qkv.QDim, h) {
			return nil, true, fmt.Errorf("verifier batch layer %d O projection rejected", l)
		}
		if layer.PreFFNNorm != nil {
			for b := 0; b < B; b++ {
				o := bOOut[b*h : (b+1)*h]
				rmsNormInPlace(o, layer.PostNorm.Data(), eps)
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], o)
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				in := bMlpIn[b*h : (b+1)*h]
				copy(in, bHidden[b*h:(b+1)*h])
				if isGemma {
					simd.RMSNormBF16(in, layer.PreFFNNorm.Data(), eps)
				} else {
					rmsNormInPlace(in, layer.PreFFNNorm.Data(), eps)
				}
			}
		} else {
			for b := 0; b < B; b++ {
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], bOOut[b*h:(b+1)*h])
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				rmsNormInPlace(bHidden[b*h:(b+1)*h], layer.PostNorm.Data(), eps)
			}
			copy(bMlpIn, bHidden)
		}
		inter := m.layerInterFor(layer)
		if !m.projBatch(bGate[:B*inter], bMlpIn[:B*h], B, layer.GateW, layer.GateWm, h, inter) || !m.projBatch(bUp[:B*inter], bMlpIn[:B*h], B, layer.UpW, layer.UpWm, h, inter) {
			return nil, true, fmt.Errorf("verifier batch layer %d MLP gate/up rejected", l)
		}
		for b := 0; b < B; b++ {
			gate := bGate[b*inter : (b+1)*inter]
			up := bUp[b*inter : (b+1)*inter]
			if isGemma {
				simd.ToBF16(gate)
				simd.ToBF16(up)
			}
			if m.Config.HiddenAct == "gelu_pytorch_tanh" {
				simd.GELUTanhMul(gate, gate, up)
				if isGemma {
					simd.ToBF16(gate)
				}
			} else {
				simd.VecSiLUMul(gate, gate, up)
			}
		}
		if !m.projBatch(bDown[:B*h], bGate[:B*inter], B, layer.DownW, layer.DownWm, inter, h) {
			return nil, true, fmt.Errorf("verifier batch layer %d MLP down rejected", l)
		}
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
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = append([]float32(nil), bHidden[b*h:(b+1)*h]...)
	}
	return out, true, nil
}

func (m *LlamaModel) mtpVerifierBatchLayerEligible(batch MTPVerifierBatchInputs) bool {
	// Keep full-layer batch lowering behind a correctness gate until the row-loop
	// parity suite is complete. The QKV projection primitive is tested and ready,
	// but the complete attention+MLP layer replacement must not silently change
	// verifier acceptance.
	if !mtpVerifierBatchLayerLoweringEnabled() || batch.HasPerLayerInputs || m == nil || m.Config.NumLayers <= 0 || m.Config.HiddenSize <= 0 || m.Config.NumHeads <= 0 || m.Config.HeadDim <= 0 {
		return false
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		if !layer.HasKV || layer.IsMoE || layer.QWq != nil || layer.KWq != nil || layer.VWq != nil || layer.OWq != nil || layer.GateWq != nil || layer.UpWq != nil || layer.DownWq != nil {
			return false
		}
		if layer.QW == nil && layer.QWm == nil || layer.KW == nil && layer.KWm == nil || layer.OW == nil && layer.OWm == nil || layer.GateW == nil && layer.GateWm == nil || layer.UpW == nil && layer.UpWm == nil || layer.DownW == nil && layer.DownWm == nil {
			return false
		}
		if !(m.Config.AttentionKEqV && ((layer.VW != nil && layer.VW == layer.KW) || (layer.VWm != nil && layer.VWm == layer.KWm))) && layer.VW == nil && layer.VWm == nil {
			return false
		}
		if layer.InputNorm == nil || layer.PostNorm == nil {
			return false
		}
	}
	return true
}

func mtpVerifierBatchLayerLoweringEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (m *LlamaModel) validateMTPVerifierBatchForwardInputs(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) error {
	if err := validateMTPVerifierPlanForModel(m, batch.Plan); err != nil {
		return err
	}
	if len(batch.HiddenRows) != len(batch.Plan.VerifierTokens) {
		return fmt.Errorf("MTP verifier batch hidden rows=%d, want verifier tokens=%d", len(batch.HiddenRows), len(batch.Plan.VerifierTokens))
	}
	for i, row := range batch.HiddenRows {
		if len(row) != m.Config.HiddenSize {
			return fmt.Errorf("MTP verifier batch hidden row %d len=%d, want %d", i, len(row), m.Config.HiddenSize)
		}
	}
	if batch.HasPerLayerInputs && len(batch.PerLayerInputs) != len(batch.Plan.VerifierTokens) {
		return fmt.Errorf("MTP verifier batch PLI rows=%d, want verifier tokens=%d", len(batch.PerLayerInputs), len(batch.Plan.VerifierTokens))
	}
	if err := batch.Attention.ValidateAgainst(batch.Plan, m); err != nil {
		return err
	}
	if len(kvCacheK) != m.Config.NumLayers || len(kvCacheV) != m.Config.NumLayers {
		return fmt.Errorf("KV cache layers K/V=%d/%d, want %d", len(kvCacheK), len(kvCacheV), m.Config.NumLayers)
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		kvDim, err := m.LayerKVDim(l)
		if err != nil {
			return err
		}
		if kvDim == 0 {
			if layer.HasKV {
				return fmt.Errorf("layer %d has invalid zero KV dim", l)
			}
			if layer.KVSourceLayer < 0 || layer.KVSourceLayer >= m.Config.NumLayers {
				return fmt.Errorf("shared-KV layer %d source %d out of range [0,%d)", l, layer.KVSourceLayer, m.Config.NumLayers)
			}
			sourceDim, err := m.LayerKVDim(layer.KVSourceLayer)
			if err != nil {
				return err
			}
			if sourceDim == 0 {
				return fmt.Errorf("shared-KV layer %d source %d does not append KV", l, layer.KVSourceLayer)
			}
			if len(kvCacheK[l]) != 0 || len(kvCacheV[l]) != 0 {
				return fmt.Errorf("shared/non-KV layer %d owns unexpected K/V cache entries %d/%d", l, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			continue
		}
		want, ok := checkedProduct(batch.Plan.StartPos, kvDim)
		if !ok {
			return fmt.Errorf("verifier KV history length overflows for layer %d start=%d kvDim=%d", l, batch.Plan.StartPos, kvDim)
		}
		if len(kvCacheK[l]) != want || len(kvCacheV[l]) != want {
			return fmt.Errorf("layer %d KV history K/V=%d/%d, want %d for start position %d", l, len(kvCacheK[l]), len(kvCacheV[l]), want, batch.Plan.StartPos)
		}
	}
	return nil
}
