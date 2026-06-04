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
	if plan.SavedCompressedKVBytes != plan.FullCompressedKVBytes-plan.EstimatedCompressedKVBytes || plan.CompressedKVRatio <= 0 || plan.CompressedKVRatio > 1 {
		t.Fatalf("bad runtime savings: %+v", plan)
	}
	if plan.EstimatedScratchBytes != 0 || plan.EstimatedTotalBytes != plan.FloatKVBytesAllocated+plan.EstimatedCompressedKVBytes+plan.EstimatedScratchBytes {
		t.Fatalf("bad runtime scratch/total: %+v", plan)
	}
	caps := kv.RuntimeTurboQuantCapabilities()
	if plan.SIMDArch != caps.Arch || plan.SIMDRotation != caps.Rotation || plan.SIMDVec != caps.Vec || plan.SIMDAVX2 != caps.AVX2 || plan.SIMDNEON != caps.NEON || plan.SIMDRVv != caps.RVV {
		t.Fatalf("bad runtime SIMD fields: %+v caps=%+v", plan, caps)
	}
}

func TestGGUFGenerationKVRuntimePlanEstimatesUnprotectedScratch(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 8, NumKVHeads: 1, HeadDim: 8, MaxSeqLen: 16}}
	plan, err := m.GenerationKVRuntimePlan(4, 4, GGUFGenerationOptions{CacheTypeK: "turbo4", CacheTypeV: "turbo2", KVResidualWindow: 2})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EstimatedScratchBytes <= 0 || plan.EstimatedTotalBytes != plan.FloatKVBytesAllocated+plan.EstimatedCompressedKVBytes+plan.EstimatedScratchBytes {
		t.Fatalf("bad unprotected scratch estimate: %+v", plan)
	}
}

func TestGGUFTurboQuantPlanSavings(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 8, NumKVHeads: 1, HeadDim: 8, MaxSeqLen: 16}}
	plan, err := m.TurboQuantPlan("turbo4", "turbo2", 2)
	if err != nil {
		t.Fatal(err)
	}
	if plan.EstimatedSavedKVBytes <= 0 || plan.EstimatedKVRatio <= 0 || plan.EstimatedKVRatio >= 1 {
		t.Fatalf("bad savings in plan: %+v", plan)
	}
}

func TestNewGGUFGenerationForwardStateUsesRuntimePlan(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 5, NumKVHeads: 1, HeadDim: 4, MaxSeqLen: 16, VocabSize: 8, HiddenSize: 4, NumHeads: 1, SSMInnerSize: 16, SSMStateSize: 4, FullAttentionInterval: 2, AttentionKeyLength: 8, AttentionValueLength: 8}}
	st, kvK, kvV, plan, err := m.newGGUFGenerationForwardState(3, 2, GGUFGenerationOptions{CacheTypeK: "turbo4", CacheTypeV: "turbo2", KVResidualWindow: 1})
	if err != nil {
		t.Fatal(err)
	}
	if st == nil || len(st.compressedKV) != 5 || plan.MaxSeq != 5 || plan.CompressedKVLayers != 2 || plan.FloatKVLayers != 3 {
		t.Fatalf("bad generation state/plan st=%v plan=%+v", st != nil, plan)
	}
	for i := 0; i < 5; i++ {
		if st.compressedKV[i] != nil {
			if len(kvK[i]) != 0 || len(kvV[i]) != 0 {
				t.Fatalf("compressed layer %d should not allocate float KV", i)
			}
			continue
		}
		if len(kvK[i]) != plan.MaxSeq*4 || len(kvV[i]) != plan.MaxSeq*4 {
			t.Fatalf("float layer %d KV lens=%d/%d", i, len(kvK[i]), len(kvV[i]))
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
