package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// GGUFTurboQuantPlan describes the native compressed-KV cache policy for a
// GGUF model. It deliberately maps llama.cpp-style cache type names onto
// go-pherence's pure Go TurboQuant cache implementation.
type GGUFTurboQuantPlan struct {
	Enabled        bool   `json:"enabled"`
	KeyType        string `json:"key_type,omitempty"`
	ValueType      string `json:"value_type,omitempty"`
	KeyBits        int    `json:"key_bits,omitempty"`
	ValueBits      int    `json:"value_bits,omitempty"`
	ResidualWindow int    `json:"residual_window"`
	Layers         int    `json:"layers"`
	KVHeads        int    `json:"kv_heads"`
	HeadDim        int    `json:"head_dim"`
	KVDim          int    `json:"kv_dim"`
	CacheLayers    int    `json:"cache_layers"`
}

func (m *GGUFLlama) TurboQuantPlan(keyType, valueType string, residualWindow int) (GGUFTurboQuantPlan, error) {
	if m == nil {
		return GGUFTurboQuantPlan{}, fmt.Errorf("nil GGUF model")
	}
	cfg := m.Config
	if cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return GGUFTurboQuantPlan{}, fmt.Errorf("invalid GGUF KV dims layers=%d kv_heads=%d head_dim=%d", cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim)
	}
	tqCfg, enabled, err := kv.TurboQuantConfigFromCacheTypes(keyType, valueType, residualWindow)
	if err != nil {
		return GGUFTurboQuantPlan{}, err
	}
	return GGUFTurboQuantPlan{
		Enabled:        enabled,
		KeyType:        keyType,
		ValueType:      valueType,
		KeyBits:        tqCfg.KeyBits,
		ValueBits:      tqCfg.ValueBits,
		ResidualWindow: tqCfg.ResidualWindow,
		Layers:         cfg.NumLayers,
		KVHeads:        cfg.NumKVHeads,
		HeadDim:        cfg.HeadDim,
		KVDim:          cfg.NumKVHeads * cfg.HeadDim,
		CacheLayers:    cfg.GGUFCompressedKVLayerCount(),
	}, nil
}

func (m *GGUFLlama) NewTurboQuantKVCache(keyType, valueType string, residualWindow int) ([]*kv.CompressedKVCache, error) {
	plan, err := m.TurboQuantPlan(keyType, valueType, residualWindow)
	if err != nil {
		return nil, err
	}
	if !plan.Enabled {
		return nil, nil
	}
	tqCfg, _, err := kv.TurboQuantConfigFromCacheTypes(keyType, valueType, residualWindow)
	if err != nil {
		return nil, err
	}
	tq := kv.NewTurboQuantState(plan.HeadDim, plan.Layers, tqCfg)
	out := make([]*kv.CompressedKVCache, plan.Layers)
	for i := range out {
		if !m.Config.GGUFUsesCompressedKVLayer(i) {
			continue
		}
		out[i] = kv.NewCompressedKVCache(plan.KVDim, plan.KVHeads, plan.HeadDim, tq, tq.IsProtectedLayer(i))
	}
	return out, nil
}
