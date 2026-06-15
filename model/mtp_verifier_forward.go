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
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	return m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
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
