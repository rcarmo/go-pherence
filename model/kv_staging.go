package model

import (
	"fmt"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// LayerHeadDim returns the effective per-head width for a layer. Gemma4 full
// attention layers use GlobalHeadDim even when a caller built a model manually
// without populating LlamaLayer.HeadDimLocal; loaded models still set
// HeadDimLocal explicitly, so this is a contract guard for all graph paths.
func (m *LlamaModel) LayerHeadDim(layerIdx int) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("nil model")
	}
	if layerIdx < 0 || layerIdx >= len(m.Layers) {
		return 0, fmt.Errorf("layer index %d out of range [0,%d)", layerIdx, len(m.Layers))
	}
	if hd := m.Layers[layerIdx].HeadDimLocal; hd > 0 {
		return hd, nil
	}
	headDim := m.Config.HeadDim
	if layerIdx < len(m.Config.LayerTypes) && m.Config.LayerTypes[layerIdx] == "full_attention" && m.Config.GlobalHeadDim > 0 {
		headDim = m.Config.GlobalHeadDim
	}
	if headDim <= 0 {
		return 0, fmt.Errorf("layer %d head_dim=%d", layerIdx, headDim)
	}
	return headDim, nil
}

// LayerKVDim returns the per-token K/V vector width appended by one layer.
// Shared-KV layers return 0 because they reuse a source layer and do not append.
func (m *LlamaModel) LayerKVDim(layerIdx int) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("nil model")
	}
	if layerIdx < 0 || layerIdx >= len(m.Layers) {
		return 0, fmt.Errorf("layer index %d out of range [0,%d)", layerIdx, len(m.Layers))
	}
	layer := m.Layers[layerIdx]
	if !layer.HasKV {
		return 0, nil
	}
	numKVHeads := gemmacfg.LayerKVHeads(m.Config, layerIdx)
	if numKVHeads <= 0 {
		return 0, fmt.Errorf("layer %d num_key_value_heads=%d", layerIdx, numKVHeads)
	}
	headDim, err := m.LayerHeadDim(layerIdx)
	if err != nil {
		return 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if numKVHeads > maxInt/headDim {
		return 0, fmt.Errorf("layer %d kv dim overflow: heads=%d head_dim=%d", layerIdx, numKVHeads, headDim)
	}
	return numKVHeads * headDim, nil
}

// LayerKVDims returns per-layer K/V widths suitable for FloatKVCheckpoint
// keep-prefix commits. Layers that do not append K/V have dimension 0.
func (m *LlamaModel) LayerKVDims() ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	dims := make([]int, len(m.Layers))
	for i := range m.Layers {
		dim, err := m.LayerKVDim(i)
		if err != nil {
			return nil, err
		}
		dims[i] = dim
	}
	return dims, nil
}

// CommitAcceptedFloatKV keeps the accepted verifier KV prefix plus bonus token
// using this model's per-layer K/V widths.
func (m *LlamaModel) CommitAcceptedFloatKV(kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint, acceptance MTPAcceptance) error {
	dims, err := m.LayerKVDims()
	if err != nil {
		return err
	}
	return CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, dims, acceptance)
}
