package model

import "fmt"

// PrefillEmbeddingsLogits runs a caller-provided row-major prompt embedding
// matrix through the validated batched CPU prefill and returns last-position
// logits. It is a stateless parity/integration surface; generation uses
// GenerateFromEmbeddings to retain and continue the KV cache.
func (m *LlamaModel) PrefillEmbeddingsLogits(tokenIDs []int, embeddings []float32) ([]float32, error) {
	hiddenSize := 0
	if m != nil {
		hiddenSize = m.Config.HiddenSize
	}
	if m == nil || len(tokenIDs) < 2 || hiddenSize <= 0 || len(embeddings) != len(tokenIDs)*hiddenSize {
		return nil, fmt.Errorf("prefill embeddings logits: tokens=%d values=%d hidden=%d", len(tokenIDs), len(embeddings), hiddenSize)
	}
	if !m.prefillCPUEligible(len(tokenIDs)) {
		return nil, fmt.Errorf("prefill embeddings logits: decoder graph is not eligible for batched CPU prefill")
	}
	kvK := make([][]float32, m.Config.NumLayers)
	kvV := make([][]float32, m.Config.NumLayers)
	for layer := range kvK {
		dim, err := m.LayerKVDim(layer)
		if err != nil {
			return nil, err
		}
		capacity, ok := checkedProduct(len(tokenIDs), dim)
		if !ok {
			return nil, fmt.Errorf("prefill embeddings logits: layer %d KV size overflow", layer)
		}
		kvK[layer] = make([]float32, 0, capacity)
		kvV[layer] = make([]float32, 0, capacity)
	}
	hidden, ok := m.prefillCPUEmbeddings(embeddings, len(tokenIDs), kvK, kvV)
	if !ok {
		return nil, fmt.Errorf("prefill embeddings logits: batched execution failed")
	}
	_, logits, _, err := m.FinishCPUDecodeStep(hidden)
	return logits, err
}
