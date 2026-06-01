package model

import "testing"

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
