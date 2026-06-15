package model

import "fmt"

// RunMTPVerifierBatchForward is the verifier-batch execution entry point. The
// first implemented lowering handles the no-layer/tail-only graph directly from
// MTPVerifierBatchInputs. Nonzero-layer batches still use RunMTPVerifierForward's
// sequential layer loop until the full batched verifier layer runner is landed.
func (m *LlamaModel) RunMTPVerifierBatchForward(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierBatchForwardInputs(batch, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
	if m.Config.NumLayers != 0 {
		return MTPVerifierResult{}, fmt.Errorf("MTP verifier batched layer execution not implemented for %d layers", m.Config.NumLayers)
	}
	logitsRows := make([][]float32, len(batch.HiddenRows))
	var finalActivation []float32
	for i, row := range batch.HiddenRows {
		hidden := append([]float32(nil), row...)
		activation, logits, _, err := m.FinishCPUDecodeStep(hidden)
		if err != nil {
			return MTPVerifierResult{}, fmt.Errorf("verifier batch decode finish row %d: %w", i, err)
		}
		logitsRows[i] = logits
		finalActivation = activation
	}
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
			if len(kvCacheK[l]) != 0 || len(kvCacheV[l]) != 0 {
				return fmt.Errorf("shared/non-KV layer %d owns unexpected K/V cache entries %d/%d", l, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			continue
		}
		want := batch.Plan.StartPos * kvDim
		if len(kvCacheK[l]) != want || len(kvCacheV[l]) != want {
			return fmt.Errorf("layer %d KV history K/V=%d/%d, want %d for start position %d", l, len(kvCacheK[l]), len(kvCacheV[l]), want, batch.Plan.StartPos)
		}
	}
	return nil
}
