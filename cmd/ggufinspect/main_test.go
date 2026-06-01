package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
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
}

func TestGGUFTurboQuantPlanFromInspectionRejectsPolicy(t *testing.T) {
	_, err := ggufTurboQuantPlanFromInspection(gguf.Inspection{Layers: 1, KVHeads: 1, HeadDim: 4, MaxSeqLen: 8}, "turbo9", "turbo2", 1)
	if err == nil {
		t.Fatal("expected invalid policy error")
	}
}
