package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestGGUFTurboQuantPlan(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}}
	plan, err := m.TurboQuantPlan("turbo4", "turbo2", 32)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Enabled || plan.KeyBits != 4 || plan.ValueBits != 2 || plan.ResidualWindow != 32 || plan.KVDim != 16 {
		t.Fatalf("unexpected plan: %+v", plan)
	}
	if plan.EstimatedTotalBytes != plan.EstimatedKVBytes+plan.EstimatedScratchBytes {
		t.Fatalf("unexpected scratch/total plan: %+v", plan)
	}
	caps := kv.RuntimeTurboQuantCapabilities()
	if plan.SIMDArch != caps.Arch || plan.SIMDRotation != caps.Rotation || plan.SIMDVec != caps.Vec || plan.SIMDAVX2 != caps.AVX2 || plan.SIMDNEON != caps.NEON || plan.SIMDRVv != caps.RVV {
		t.Fatalf("unexpected SIMD plan fields: %+v caps=%+v", plan, caps)
	}
	caches, err := m.NewTurboQuantKVCache("turbo4", "turbo2", 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(caches) != 4 || caches[0] == nil {
		t.Fatalf("bad caches: %+v", caches)
	}
}

func TestGGUFTurboQuantPlanFullPrecisionDisabled(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4}}
	plan, err := m.TurboQuantPlan("f16", "f16", -1)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Enabled {
		t.Fatalf("full precision should disable compressed KV: %+v", plan)
	}
	caches, err := m.NewTurboQuantKVCache("f16", "f16", -1)
	if err != nil {
		t.Fatal(err)
	}
	if caches != nil {
		t.Fatalf("expected nil caches, got %+v", caches)
	}
}
