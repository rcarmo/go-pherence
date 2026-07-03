package model

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
