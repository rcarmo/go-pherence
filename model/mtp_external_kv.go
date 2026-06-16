package model

import "fmt"

// MapMTPDrafterKVSourceLayersByWidth maps q-only drafter layers to main-model
// KV-producing layers. Gemma4 assistant mirrors llama.cpp shared memory exactly:
// all SWA assistant layers share target layer n_layer-2 and full layers share
// target layer n_layer-1. Other assistants fall back to unique width matching.
func MapMTPDrafterKVSourceLayersByWidth(m *LlamaModel, d *Gemma4MTPDrafter, seqLen int) ([]int, error) {
	if m == nil || d == nil || seqLen <= 0 {
		return nil, fmt.Errorf("invalid MTP KV source mapping inputs")
	}
	if d.Config.NumLayers < 0 || len(d.Layers) < d.Config.NumLayers {
		return nil, fmt.Errorf("invalid drafter layers=%d/%d", len(d.Layers), d.Config.NumLayers)
	}
	if m.Config.ModelType == "gemma4_text" && d.Config.ModelType == "gemma4_text" && m.Config.NumLayers >= 2 {
		swaSource := m.Config.NumLayers - 2
		fullSource := m.Config.NumLayers - 1
		sources := make([]int, d.Config.NumLayers)
		for i := 0; i < d.Config.NumLayers; i++ {
			isSWA := true
			if i < len(d.Config.LayerTypes) {
				isSWA = d.Config.LayerTypes[i] != "full_attention"
			}
			if isSWA {
				sources[i] = swaSource
			} else {
				sources[i] = fullSource
			}
		}
		return sources, nil
	}
	sources := make([]int, d.Config.NumLayers)
	used := make(map[int]bool)
	for i := 0; i < d.Config.NumLayers; i++ {
		headDim := drafterLayerHeadDim(d, i)
		kvHeads := drafterLayerKVHeads(d, i)
		if headDim <= 0 || kvHeads <= 0 {
			return nil, fmt.Errorf("invalid drafter layer %d KV dims heads=%d headDim=%d", i, kvHeads, headDim)
		}
		wantDim, ok := checkedProduct(kvHeads, headDim)
		if !ok {
			return nil, fmt.Errorf("drafter layer %d KV dim overflow", i)
		}
		best := -1
		for l := 0; l < len(m.Layers); l++ {
			if used[l] {
				continue
			}
			dim, err := m.LayerKVDim(l)
			if err != nil {
				return nil, err
			}
			if dim == wantDim {
				best = l
				break
			}
		}
		if best < 0 {
			wantLen, _ := checkedProduct(seqLen, wantDim)
			return nil, fmt.Errorf("no main KV source for drafter layer %d width=%d len=%d", i, wantDim, wantLen)
		}
		sources[i] = best
		used[best] = true
	}
	return sources, nil
}

// NewMTPDrafterExternalKVFromPromptContext builds the explicit q-only drafter
// external-KV view from a main-model prompt context using width-based layer
// mapping. It validates the resulting view before returning it.
func NewMTPDrafterExternalKVFromPromptContext(m *LlamaModel, d *Gemma4MTPDrafter, ctx MTPPromptContext) (*MTPDrafterExternalKV, error) {
	if ctx.SeqLen <= 0 || len(ctx.KVCacheK) == 0 || len(ctx.KVCacheV) != len(ctx.KVCacheK) {
		return nil, fmt.Errorf("invalid MTP prompt context KV K/V layers=%d/%d seqLen=%d", len(ctx.KVCacheK), len(ctx.KVCacheV), ctx.SeqLen)
	}
	sources, err := MapMTPDrafterKVSourceLayersByWidth(m, d, ctx.SeqLen)
	if err != nil {
		return nil, err
	}
	kvK := ctx.KVCacheK
	kvV := ctx.KVCacheV
	if m != nil && len(kvK) == len(m.Layers) && len(kvV) == len(m.Layers) {
		kvK = append([][]float32(nil), kvK...)
		kvV = append([][]float32(nil), kvV...)
		for i, layer := range m.Layers {
			if !layer.HasKV && layer.KVSourceLayer >= 0 && layer.KVSourceLayer < len(kvK) {
				kvK[i] = kvK[layer.KVSourceLayer]
				kvV[i] = kvV[layer.KVSourceLayer]
			}
		}
	}
	externalKV := &MTPDrafterExternalKV{K: kvK, V: kvV, SourceLayers: sources, SeqLen: ctx.SeqLen}
	if d != nil && d.Config.NumLayers > 0 {
		if err := validateMTPDrafterExternalKV(d, externalKV); err != nil {
			return nil, err
		}
	}
	return externalKV, nil
}
