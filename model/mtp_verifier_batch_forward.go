package model

import "fmt"

// RunMTPVerifierBatchForward is the verifier-batch execution entry point. It
// consumes materialized MTPVerifierBatchInputs for all verifier rows. The current
// lowering executes rows/layers sequentially while preserving the explicit batch
// and attention-plan contract; future SIMD/GPU work can replace the inner layer
// loop without changing callers.
func (m *LlamaModel) RunMTPVerifierBatchForward(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierBatchForwardInputs(batch, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
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
					return MTPVerifierResult{}, fmt.Errorf("verifier batch forward layer %d at position %d: %w", l, pos, err)
				}
				continue
			}
			hidden = m.ForwardLayer(hidden, l, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return MTPVerifierResult{}, fmt.Errorf("verifier batch forward layer %d at position %d failed", l, pos)
			}
		}
		finalHiddenRows[i] = hidden
	}
	finalActivations, logitsRows, _, err := m.FinishCPUDecodeBatch(finalHiddenRows)
	if err != nil {
		return MTPVerifierResult{}, fmt.Errorf("verifier batch decode finish: %w", err)
	}
	finalActivation := finalActivations[len(finalActivations)-1]
	return NewMTPVerifierResultForModel(m, batch.Plan.InputToken, batch.Plan.DraftedTokens, logitsRows, finalActivation)
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
