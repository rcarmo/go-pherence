package lfm2

import (
	"path/filepath"
	"testing"
)

func TestRuntimePlanSizing(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ConvLayers != 21 || plan.FullAttentionLayers != 3 {
		t.Fatalf("layer plan=%+v", plan)
	}
	if plan.ConvStateFloats != 129024 {
		t.Fatalf("conv state floats=%d", plan.ConvStateFloats)
	}
	if plan.KVFloatsPerToken != 3072 {
		t.Fatalf("kv floats/token=%d", plan.KVFloatsPerToken)
	}
	kvBytes, err := plan.KVBytes(128, 2)
	if err != nil {
		t.Fatal(err)
	}
	if kvBytes != 786432 {
		t.Fatalf("kv bytes=%d", kvBytes)
	}
	convBytes, err := plan.ConvStateBytes(2)
	if err != nil {
		t.Fatal(err)
	}
	if convBytes != 258048 {
		t.Fatalf("conv bytes=%d", convBytes)
	}
}

func TestRuntimePlanRejectsBadStateSizing(t *testing.T) {
	p := RuntimePlan{HiddenSize: 2048, HeadDim: 64, Layers: 2, ConvLayers: 1, FullAttentionLayers: 1, KVHeads: 8, ConvLCache: 3, ConvStateFloats: 1, KVFloatsPerToken: 1024, Experts: 32, ExpertsPerToken: 4, MoEIntermediate: 1792}
	if err := p.Validate(); err == nil {
		t.Fatal("expected conv state size error")
	}
	if _, err := p.KVBytes(-1, 2); err == nil {
		t.Fatal("expected invalid KV sizing error")
	}
	if _, err := p.ConvStateBytes(0); err == nil {
		t.Fatal("expected invalid conv sizing error")
	}
}

func TestRuntimePlanAppliesStateLayoutContracts(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ConvStateFloats != plan.ConvStateLayout.TotalFloats {
		t.Fatalf("conv state contract not applied: plan=%d layout=%d", plan.ConvStateFloats, plan.ConvStateLayout.TotalFloats)
	}
	if plan.KVFloatsPerToken != plan.AttentionKVLayout.FloatsPerToken {
		t.Fatalf("KV contract not applied: plan=%d layout=%d", plan.KVFloatsPerToken, plan.AttentionKVLayout.FloatsPerToken)
	}
	plan.ConvStateFloats++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected conv state layout mismatch")
	}
	plan.ConvStateFloats = plan.ConvStateLayout.TotalFloats
	plan.KVFloatsPerToken++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected attention KV layout mismatch")
	}
}
