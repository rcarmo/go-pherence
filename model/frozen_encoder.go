package model

import "fmt"

// EncodeTokenHiddenStates runs tokenIDs through the decoder and returns one
// final-normalized causal hidden-state row per token. It owns its KV cache and
// does not compute logits.
func (m *LlamaModel) EncodeTokenHiddenStates(tokenIDs []int) ([][]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if err := m.validateFrozenEncoderSupport(); err != nil {
		return nil, err
	}
	if len(tokenIDs) > 1 && m.prefillCPUEligible(len(tokenIDs)) {
		if hiddenRows, ok, err := m.encodeTokenHiddenStatesPrefill(tokenIDs); err != nil {
			return nil, err
		} else if ok {
			return hiddenRows, nil
		}
	}
	return m.encodeTokenHiddenStatesSequential(tokenIDs)
}

func (m *LlamaModel) validateFrozenEncoderSupport() error {
	cfg := m.Config
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("invalid model layer count %d", cfg.NumLayers)
	}
	if len(m.Layers) < cfg.NumLayers {
		return fmt.Errorf("model has %d layers, want at least %d", len(m.Layers), cfg.NumLayers)
	}
	if cfg.ModelType == "gemma4_text" || m.PerLayerModelProj != nil {
		return fmt.Errorf("EncodeTokenHiddenStates does not support gemma4/per-layer input models")
	}
	if cfg.NumExperts > 0 {
		return fmt.Errorf("EncodeTokenHiddenStates does not support mixture-of-experts models")
	}
	for i := 0; i < cfg.NumLayers; i++ {
		layer := m.Layers[i]
		if layer.PLIGate != nil || layer.PLIProj != nil {
			return fmt.Errorf("EncodeTokenHiddenStates does not support gemma4/per-layer input models")
		}
		if layer.IsMoE || layer.RouterW != nil || len(layer.ExpertGateW) > 0 || len(layer.ExpertUpW) > 0 || len(layer.ExpertDownW) > 0 {
			return fmt.Errorf("EncodeTokenHiddenStates does not support mixture-of-experts layer %d", i)
		}
	}
	return nil
}

func (m *LlamaModel) encodeTokenHiddenStatesPrefill(tokenIDs []int) ([][]float32, bool, error) {
	B := len(tokenIDs)
	h := m.Config.HiddenSize
	batchHidden, ok := checkedProduct(B, h)
	if !ok {
		return nil, false, nil
	}
	bHidden := make([]float32, batchHidden)
	for i, tokID := range tokenIDs {
		if err := m.ScaledTokenEmbeddingInto(bHidden[i*h:(i+1)*h], tokID); err != nil {
			return nil, false, fmt.Errorf("token %d embedding: %w", i, err)
		}
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	hiddenRows, ok := m.prefillCPUHiddenAll(bHidden, B, kvCacheK, kvCacheV)
	if !ok {
		return nil, false, nil
	}
	for i := range hiddenRows {
		if err := m.finalizeCPUHidden(hiddenRows[i]); err != nil {
			return nil, false, fmt.Errorf("token %d final norm: %w", i, err)
		}
	}
	return hiddenRows, true, nil
}

func (m *LlamaModel) encodeTokenHiddenStatesSequential(tokenIDs []int) ([][]float32, error) {
	h := m.Config.HiddenSize
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	hiddenRows := make([][]float32, len(tokenIDs))
	for pos, tokID := range tokenIDs {
		hidden := make([]float32, h)
		if err := m.ScaledTokenEmbeddingInto(hidden, tokID); err != nil {
			return nil, fmt.Errorf("token %d embedding: %w", pos, err)
		}
		for layerIdx := 0; layerIdx < m.Config.NumLayers; layerIdx++ {
			hidden = m.ForwardLayer(hidden, layerIdx, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return nil, fmt.Errorf("token %d layer %d forward failed", pos, layerIdx)
			}
		}
		if err := m.finalizeCPUHidden(hidden); err != nil {
			return nil, fmt.Errorf("token %d final norm: %w", pos, err)
		}
		hiddenRows[pos] = append([]float32(nil), hidden...)
	}
	return hiddenRows, nil
}
