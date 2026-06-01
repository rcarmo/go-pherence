package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestGGUFCompressedKVLayersPlainModel(t *testing.T) {
	cfg := GGUFLlamaConfig{NumLayers: 3}
	if got := cfg.GGUFCompressedKVLayerCount(); got != 3 {
		t.Fatalf("plain layer count=%d", got)
	}
	for i := 0; i < 3; i++ {
		if !cfg.GGUFUsesCompressedKVLayer(i) {
			t.Fatalf("plain layer %d should use KV", i)
		}
	}
}

func TestGGUFCompressedKVLayersQwenNextInterval(t *testing.T) {
	cfg := GGUFLlamaConfig{NumLayers: 10, SSMInnerSize: 16, SSMStateSize: 4, FullAttentionInterval: 4, AttentionKeyLength: 8, AttentionValueLength: 8}
	if got := cfg.GGUFCompressedKVLayerCount(); got != 2 {
		t.Fatalf("qwennext layer count=%d", got)
	}
	want := map[int]bool{3: true, 7: true}
	for i := 0; i < cfg.NumLayers; i++ {
		if got := cfg.GGUFUsesCompressedKVLayer(i); got != want[i] {
			t.Fatalf("layer %d compressed=%v want %v", i, got, want[i])
		}
	}
}

func TestGGUFTurboQuantKVBytesEstimatesCompressedCache(t *testing.T) {
	cfg := GGUFLlamaConfig{NumLayers: 4, NumKVHeads: 1, HeadDim: 8, MaxSeqLen: 10}
	full, estimated := cfg.GGUFTurboQuantKVBytes(kv.TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 2}, true)
	// Full precision: layers * tokens * kvDim * K/V * sizeof(float32).
	if full != 4*10*8*2*4 {
		t.Fatalf("full bytes=%d", full)
	}
	if estimated <= 0 || estimated >= full {
		t.Fatalf("estimated bytes=%d full=%d", estimated, full)
	}
}

func TestGGUFGenerationKVRuntimePlan(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 5, NumKVHeads: 1, HeadDim: 4, MaxSeqLen: 16, SSMInnerSize: 16, SSMStateSize: 4, FullAttentionInterval: 2, AttentionKeyLength: 8, AttentionValueLength: 8}}
	plan, err := m.GenerationKVRuntimePlan(3, 2, GGUFGenerationOptions{CacheTypeK: "turbo4", CacheTypeV: "turbo2", KVResidualWindow: 1})
	if err != nil {
		t.Fatal(err)
	}
	if plan.MaxSeq != 5 || plan.CompressedKVLayers != 2 || plan.FloatKVLayers != 3 {
		t.Fatalf("bad runtime plan: %+v", plan)
	}
	if want := int64(3 * 5 * 4 * 2 * 4); plan.FloatKVBytesAllocated != want {
		t.Fatalf("float bytes=%d want %d", plan.FloatKVBytesAllocated, want)
	}
	if plan.FullCompressedKVBytes <= 0 || plan.EstimatedCompressedKVBytes <= 0 || plan.EstimatedCompressedKVBytes > plan.FullCompressedKVBytes {
		t.Fatalf("bad compressed bytes: %+v", plan)
	}
}

func TestNewTurboQuantKVCacheSkipsQwenNextRecurrentLayers(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 5, NumKVHeads: 1, HeadDim: 4, SSMInnerSize: 16, SSMStateSize: 4, FullAttentionInterval: 2, AttentionKeyLength: 8, AttentionValueLength: 8}}
	caches, err := m.NewTurboQuantKVCache("turbo4", "turbo2", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(caches) != 5 {
		t.Fatalf("cache len=%d", len(caches))
	}
	for i := range caches {
		wantNil := (i+1)%2 != 0
		if (caches[i] == nil) != wantNil {
			t.Fatalf("layer %d cache nil=%v want nil=%v", i, caches[i] == nil, wantNil)
		}
	}
}
