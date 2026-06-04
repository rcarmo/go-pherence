package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestGGUFTurboQuantPlanFromInspection(t *testing.T) {
	in := gguf.Inspection{RuntimeSupported: true, Layers: 40, KVHeads: 2, HeadDim: 256, KVDim: 512, MaxSeqLen: 262144, FullAttentionInterval: 4, CompressedKVLayers: 10}
	plan, err := ggufTurboQuantPlanFromInspection(in, "turbo4", "turbo2", 128)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Enabled || !plan.RuntimeReady || plan.KeyBits != 4 || plan.ValueBits != 2 || plan.CacheLayers != 10 || plan.ProtectedLayers != 1 {
		t.Fatalf("bad plan: %+v", plan)
	}
	if plan.FullKVBytes != 10737418240 || plan.EstimatedBytes != 2059448320 || plan.SavedBytes != 8677969920 || plan.EstimatedKVRatio <= 0 || plan.EstimatedKVRatio >= 1 {
		t.Fatalf("bad byte estimate: %+v", plan)
	}
	if plan.EstimatedScratchBytes != 9663699456 || plan.EstimatedTotalBytes != 11723147776 {
		t.Fatalf("bad scratch/total estimate: %+v", plan)
	}
	caps := kv.RuntimeTurboQuantCapabilities()
	if plan.SIMDArch != caps.Arch || plan.SIMDRotation != caps.Rotation || plan.SIMDVec != caps.Vec || plan.SIMDAVX2 != caps.AVX2 || plan.SIMDNEON != caps.NEON || plan.SIMDRVv != caps.RVV {
		t.Fatalf("bad SIMD plan fields: %+v caps=%+v", plan, caps)
	}
}

func TestGGUFTurboQuantPlanFromInspectionRejectsPolicy(t *testing.T) {
	_, err := ggufTurboQuantPlanFromInspection(gguf.Inspection{Layers: 1, KVHeads: 1, HeadDim: 4, MaxSeqLen: 8}, "turbo9", "turbo2", 1)
	if err == nil {
		t.Fatal("expected invalid policy error")
	}
}
