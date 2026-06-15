package model

import "fmt"

// MapMTPDrafterKVSourceLayersByWidth maps q-only drafter layers to main-model
// KV-producing layers with matching per-token K/V width. Each main source layer
// is used at most once, matching the current Gemma4 assistant/shared-KV layout.
func MapMTPDrafterKVSourceLayersByWidth(m *LlamaModel, d *Gemma4MTPDrafter, seqLen int) ([]int, error) {
	if m == nil || d == nil || seqLen <= 0 {
		return nil, fmt.Errorf("invalid MTP KV source mapping inputs")
	}
	if d.Config.NumLayers < 0 || len(d.Layers) < d.Config.NumLayers {
		return nil, fmt.Errorf("invalid drafter layers=%d/%d", len(d.Layers), d.Config.NumLayers)
	}
	sources := make([]int, d.Config.NumLayers)
	used := make(map[int]bool)
	for i := 0; i < d.Config.NumLayers; i++ {
		headDim := d.Config.HeadDim
		if d.Layers[i].HeadDimLocal > 0 {
			headDim = d.Layers[i].HeadDimLocal
		}
		kvHeads := d.Config.NumKVHeads
		if i < len(d.Config.LayerTypes) && d.Config.LayerTypes[i] == "full_attention" && d.Config.NumGlobalKVHeads > 0 {
			kvHeads = d.Config.NumGlobalKVHeads
		}
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
	externalKV := &MTPDrafterExternalKV{K: ctx.KVCacheK, V: ctx.KVCacheV, SourceLayers: sources, SeqLen: ctx.SeqLen}
	if d != nil && d.Config.NumLayers > 0 {
		if err := validateMTPDrafterExternalKV(d, externalKV); err != nil {
			return nil, err
		}
	}
	return externalKV, nil
}
