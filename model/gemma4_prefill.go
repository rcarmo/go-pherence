package model

import "fmt"

// legacyCPUPrefillBatchEligible keeps the exact token-at-a-time path as the
// fallback for state/features not covered by the layer-batched lowering.
func (m *LlamaModel) legacyCPUPrefillBatchEligible(st *cpuTokenState, batch int) bool {
	if m == nil || st == nil || batch < 8 || st.compressedKV != nil {
		return false
	}
	if debugOpHook != nil || debugLayerHook != nil || debugCPUPerLayerInputsOverrideHook != nil || debugCPUHiddenInOverrideHook != nil || debugCPUMLPInputOverrideHook != nil || debugLogitsHook != nil {
		return false
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		// runLegacyCPUToken does not execute GGUF-only PLI gates. Dense PLI and
		// MoE retain the authoritative sequential implementation.
		if layer.IsMoE || layer.PLIGate != nil {
			return false
		}
	}
	return true
}

// runLegacyCPUPrefillBatch lowers a contiguous prompt range layer-by-layer.
// The batch carrier is deliberately built from the legacy token contract: it
// uses scaled token embeddings, exact positions/causal ranges, request-owned
// float KV, and no GGUF-only PLI branch. The layer implementation preserves the
// same per-row attention and reduction semantics as runLegacyCPUToken.
func (m *LlamaModel) runLegacyCPUPrefillBatch(st *cpuTokenState, tokens []int, startPos int) ([]float32, bool, error) {
	if !m.legacyCPUPrefillBatchEligible(st, len(tokens)) {
		return nil, false, nil
	}
	rows, err := m.runLegacyCPUPrefillBatchLayers(st, tokens, startPos)
	if err != nil {
		return nil, true, err
	}
	if len(rows) != len(tokens) {
		return nil, true, fmt.Errorf("legacy CPU prefill rows=%d, want %d", len(rows), len(tokens))
	}
	copy(st.hidden, rows[len(rows)-1])
	return st.hidden, true, nil
}

func (m *LlamaModel) runLegacyCPUPrefillBatchLayers(st *cpuTokenState, tokens []int, startPos int) ([][]float32, error) {
	if m == nil || st == nil || len(tokens) == 0 {
		return nil, fmt.Errorf("legacy CPU prefill invalid model/state/tokens")
	}
	B, h := len(tokens), m.Config.HiddenSize
	plan := MTPVerifierPlan{
		InputToken:     tokens[0],
		DraftedTokens:  append([]int(nil), tokens[1:]...),
		VerifierTokens: append([]int(nil), tokens...),
		StartPos:       startPos,
		Positions:      make([]int, B),
	}
	hiddenFlat := make([]float32, B*h)
	hiddenRows := make([][]float32, B)
	for i, tok := range tokens {
		plan.Positions[i] = startPos + i
		hiddenRows[i] = hiddenFlat[i*h : (i+1)*h]
		if err := m.ScaledTokenEmbeddingInto(hiddenRows[i], tok); err != nil {
			return nil, fmt.Errorf("legacy CPU prefill token %d embedding: %w", startPos+i, err)
		}
	}
	attention, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		return nil, fmt.Errorf("legacy CPU prefill attention plan: %w", err)
	}
	batch := MTPVerifierBatchInputs{
		Plan:           plan,
		HiddenFlat:     hiddenFlat,
		HiddenRows:     hiddenRows,
		PerLayerInputs: make([][][]float32, B),
		Attention:      attention,
	}
	batch.Scratch, err = NewMTPVerifierBatchScratchPlan(m, batch)
	if err != nil {
		return nil, fmt.Errorf("legacy CPU prefill scratch plan: %w", err)
	}
	rows, ok, err := m.runMTPVerifierBatchLayers(batch, st.kvCacheK, st.kvCacheV)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("legacy CPU prefill layer lowering unsupported")
	}
	return rows, nil
}
