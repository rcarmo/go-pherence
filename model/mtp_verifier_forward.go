package model

import "fmt"

// RunMTPVerifierForward runs a CPU verifier pass over plan.VerifierTokens at
// plan.Positions, writing candidate K/V into the provided staged float caches
// and returning per-position logits plus the final verifier activation.
//
// Current contract: float KV only. kvCacheK/V must already contain exactly
// plan.StartPos prompt/history tokens for every layer that appends K/V. Gemma4
// per-layer-input gating uses the same per-token helper and layer path as prompt
// context construction.
func (m *LlamaModel) RunMTPVerifierForward(plan MTPVerifierPlan, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierForwardInputs(plan, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if m.Config.NumLayers == 0 {
		return m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	}
	logitsRows := make([][]float32, len(plan.VerifierTokens))
	var finalActivation []float32
	maxSeqLen := plan.StartPos + len(plan.VerifierTokens)
	if maxSeqLen < 1 {
		maxSeqLen = 1
	}
	attnScoresScratch := make([]float32, maxSeqLen)
	maxHeadDim := m.Config.HeadDim
	for i := range m.Layers {
		if m.Layers[i].HeadDimLocal > maxHeadDim {
			maxHeadDim = m.Layers[i].HeadDimLocal
		}
	}
	attnOutScratch := make([]float32, m.Config.NumHeads*maxHeadDim)
	for i := range batch.Plan.VerifierTokens {
		hidden := append([]float32(nil), batch.HiddenRows[i]...)
		pos := batch.Plan.Positions[i]
		perLayerInputs := batch.PerLayerInputs[i]
		for l := 0; l < m.Config.NumLayers; l++ {
			if perLayerInputs != nil {
				hidden, err = m.forwardMTPPromptLayer(hidden, perLayerInputs, l, pos, kvCacheK, kvCacheV, attnScoresScratch, attnOutScratch)
				if err != nil {
					return MTPVerifierResult{}, fmt.Errorf("verifier forward layer %d at position %d: %w", l, pos, err)
				}
				continue
			}
			hidden = m.ForwardLayer(hidden, l, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return MTPVerifierResult{}, fmt.Errorf("verifier forward layer %d at position %d failed", l, pos)
			}
		}
		activation, logits, _, err := m.finishCPUDecodeStep(hidden)
		if err != nil {
			return MTPVerifierResult{}, fmt.Errorf("verifier decode finish at position %d: %w", pos, err)
		}
		logitsRows[i] = logits
		finalActivation = activation
	}
	return NewMTPVerifierResultForModel(m, batch.Plan.InputToken, batch.Plan.DraftedTokens, logitsRows, finalActivation)
}

func (m *LlamaModel) validateMTPVerifierForwardInputs(plan MTPVerifierPlan, kvCacheK, kvCacheV [][]float32) error {
	if err := validateMTPVerifierPlanForModel(m, plan); err != nil {
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
				return fmt.Errorf("shared-KV layer %d owns unexpected K/V cache entries %d/%d", l, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			continue
		}
		want, ok := checkedProduct(plan.StartPos, kvDim)
		if !ok {
			return fmt.Errorf("verifier KV history length overflows for layer %d start=%d kvDim=%d", l, plan.StartPos, kvDim)
		}
		if len(kvCacheK[l]) != want || len(kvCacheV[l]) != want {
			return fmt.Errorf("layer %d KV history K/V=%d/%d, want %d for start position %d", l, len(kvCacheK[l]), len(kvCacheV[l]), want, plan.StartPos)
		}
	}
	return nil
}
