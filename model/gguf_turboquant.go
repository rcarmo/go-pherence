package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// GGUFTurboQuantPlan describes the native compressed-KV cache policy for a
// GGUF model. It deliberately maps llama.cpp-style cache type names onto
// go-pherence's pure Go TurboQuant cache implementation.
type GGUFGenerationKVRuntimePlan struct {
	MaxSeq                     int     `json:"max_seq"`
	FloatKVLayers              int     `json:"float_kv_layers"`
	CompressedKVLayers         int     `json:"compressed_kv_layers"`
	ProtectedCompressedLayers  int     `json:"protected_compressed_layers"`
	FloatKVBytesAllocated      int64   `json:"float_kv_bytes_allocated"`
	FullCompressedKVBytes      int64   `json:"full_compressed_kv_bytes"`
	EstimatedCompressedKVBytes int64   `json:"estimated_compressed_kv_bytes"`
	SavedCompressedKVBytes     int64   `json:"saved_compressed_kv_bytes"`
	CompressedKVRatio          float64 `json:"compressed_kv_ratio"`
	EstimatedScratchBytes      int64   `json:"estimated_scratch_bytes,omitempty"`
	EstimatedTotalBytes        int64   `json:"estimated_total_bytes,omitempty"`
	SIMDArch                   string  `json:"simd_arch"`
	SIMDRotation               bool    `json:"simd_rotation"`
	SIMDVec                    bool    `json:"simd_vec"`
	SIMDAVX2                   bool    `json:"simd_avx2"`
	SIMDNEON                   bool    `json:"simd_neon"`
	SIMDRVv                    bool    `json:"simd_rvv"`
}

type GGUFTurboQuantPlan struct {
	Enabled               bool    `json:"enabled"`
	KeyType               string  `json:"key_type,omitempty"`
	ValueType             string  `json:"value_type,omitempty"`
	KeyBits               int     `json:"key_bits,omitempty"`
	ValueBits             int     `json:"value_bits,omitempty"`
	ResidualWindow        int     `json:"residual_window"`
	Layers                int     `json:"layers"`
	KVHeads               int     `json:"kv_heads"`
	HeadDim               int     `json:"head_dim"`
	KVDim                 int     `json:"kv_dim"`
	CacheLayers           int     `json:"cache_layers"`
	ProtectedCacheLayers  int     `json:"protected_cache_layers,omitempty"`
	MaxSeqLen             int     `json:"max_seq_len,omitempty"`
	FullKVBytes           int64   `json:"full_kv_bytes,omitempty"`
	EstimatedKVBytes      int64   `json:"estimated_kv_bytes,omitempty"`
	EstimatedSavedKVBytes int64   `json:"estimated_saved_kv_bytes,omitempty"`
	EstimatedKVRatio      float64 `json:"estimated_kv_ratio,omitempty"`
	SIMDArch              string  `json:"simd_arch"`
	SIMDRotation          bool    `json:"simd_rotation"`
	SIMDVec               bool    `json:"simd_vec"`
	SIMDAVX2              bool    `json:"simd_avx2"`
	SIMDNEON              bool    `json:"simd_neon"`
	SIMDRVv               bool    `json:"simd_rvv"`
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
	plan := GGUFTurboQuantPlan{
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
		MaxSeqLen:      cfg.MaxSeqLen,
	}
	est := kv.EstimateTurboQuantKV(cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.MaxSeqLen, tqCfg, enabled, cfg.GGUFUsesCompressedKVLayer)
	plan.FullKVBytes = est.FullBytes
	plan.EstimatedKVBytes = est.EstimatedBytes
	plan.EstimatedSavedKVBytes = est.SavedBytes
	plan.EstimatedKVRatio = est.Ratio
	plan.ProtectedCacheLayers = est.ProtectedLayers
	caps := kv.RuntimeTurboQuantCapabilities()
	plan.SIMDArch = caps.Arch
	plan.SIMDRotation = caps.Rotation
	plan.SIMDVec = caps.Vec
	plan.SIMDAVX2 = caps.AVX2
	plan.SIMDNEON = caps.NEON
	plan.SIMDRVv = caps.RVV
	return plan, nil
}

func (m *GGUFLlama) GenerationKVRuntimePlan(promptLen, maxNew int, opts GGUFGenerationOptions) (GGUFGenerationKVRuntimePlan, error) {
	if m == nil {
		return GGUFGenerationKVRuntimePlan{}, fmt.Errorf("nil GGUF model")
	}
	cfg := m.Config
	if promptLen < 0 || maxNew < 0 {
		return GGUFGenerationKVRuntimePlan{}, fmt.Errorf("invalid generation lengths prompt=%d max_new=%d", promptLen, maxNew)
	}
	maxSeq := promptLen + maxNew
	if maxSeq > cfg.MaxSeqLen {
		maxSeq = cfg.MaxSeqLen
	}
	if maxSeq < 0 {
		maxSeq = 0
	}
	tqCfg := kv.DefaultTurboQuantConfig()
	enabled := false
	if opts.CacheTypeK != "" || opts.CacheTypeV != "" || opts.KVResidualWindow >= 0 {
		var err error
		tqCfg, enabled, err = kv.TurboQuantConfigFromCacheTypes(opts.CacheTypeK, opts.CacheTypeV, opts.KVResidualWindow)
		if err != nil {
			return GGUFGenerationKVRuntimePlan{}, err
		}
	}
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	plan := GGUFGenerationKVRuntimePlan{MaxSeq: maxSeq}
	for i := 0; i < cfg.NumLayers; i++ {
		if enabled && cfg.GGUFUsesCompressedKVLayer(i) {
			plan.CompressedKVLayers++
			continue
		}
		plan.FloatKVLayers++
	}
	plan.FloatKVBytesAllocated = int64(plan.FloatKVLayers) * int64(maxSeq) * int64(kvDim) * 2 * 4
	est := kv.EstimateTurboQuantKV(cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, maxSeq, tqCfg, enabled, cfg.GGUFUsesCompressedKVLayer)
	plan.FullCompressedKVBytes = est.FullBytes
	plan.EstimatedCompressedKVBytes = est.EstimatedBytes
	plan.SavedCompressedKVBytes = est.SavedBytes
	plan.CompressedKVRatio = est.Ratio
	plan.ProtectedCompressedLayers = est.ProtectedLayers
	plan.EstimatedScratchBytes = est.EstimatedScratchBytes
	plan.EstimatedTotalBytes = plan.FloatKVBytesAllocated + plan.EstimatedCompressedKVBytes + plan.EstimatedScratchBytes
	caps := kv.RuntimeTurboQuantCapabilities()
	plan.SIMDArch = caps.Arch
	plan.SIMDRotation = caps.Rotation
	plan.SIMDVec = caps.Vec
	plan.SIMDAVX2 = caps.AVX2
	plan.SIMDNEON = caps.NEON
	plan.SIMDRVv = caps.RVV
	return plan, nil
}

func (m *GGUFLlama) newGGUFGenerationForwardState(promptLen, maxNew int, opts GGUFGenerationOptions) (*GGUFForwardState, [][]float32, [][]float32, GGUFGenerationKVRuntimePlan, error) {
	plan, err := m.GenerationKVRuntimePlan(promptLen, maxNew, opts)
	if err != nil {
		return nil, nil, nil, GGUFGenerationKVRuntimePlan{}, err
	}
	cfg := m.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	var compressedKV []*kv.CompressedKVCache
	if opts.CacheTypeK != "" || opts.CacheTypeV != "" || opts.KVResidualWindow >= 0 {
		compressedKV, err = m.NewTurboQuantKVCache(opts.CacheTypeK, opts.CacheTypeV, opts.KVResidualWindow)
		if err != nil {
			return nil, nil, nil, GGUFGenerationKVRuntimePlan{}, err
		}
	}
	kvK := make([][]float32, cfg.NumLayers)
	kvV := make([][]float32, cfg.NumLayers)
	for i := range kvK {
		if i < len(compressedKV) && compressedKV[i] != nil {
			continue
		}
		kvK[i] = make([]float32, plan.MaxSeq*kvDim)
		kvV[i] = make([]float32, plan.MaxSeq*kvDim)
	}
	state := m.NewForwardState()
	state.compressedKV = compressedKV
	return state, kvK, kvV, plan, nil
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
