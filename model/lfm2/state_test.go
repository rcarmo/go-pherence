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

func TestRuntimePlanAppliesScheduleExecutionContracts(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ConvLayers != len(plan.Schedule.ConvIndices) || plan.FullAttentionLayers != len(plan.Schedule.FullAttentionIndices) {
		t.Fatalf("schedule contract not applied: plan=%+v schedule=%+v", plan, plan.Schedule)
	}
	if plan.FFNLayout.DenseLayers != len(plan.Execution.DenseIndices) || plan.FFNLayout.MoELayers != len(plan.Execution.MoEIndices) {
		t.Fatalf("execution contract not applied: ffn=%+v execution=%+v", plan.FFNLayout, plan.Execution)
	}
	plan.ConvLayers++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected schedule conv-layer mismatch")
	}
	plan.ConvLayers = len(plan.Schedule.ConvIndices)
	plan.FFNLayout.MoELayers++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected execution/FFN MoE-layer mismatch")
	}
}

func TestRuntimePlanAppliesContextEmbeddingContracts(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if plan.EmbeddingLayout.VocabSize != plan.ContextLayout.VocabSize || plan.EmbeddingLayout.HiddenSize != plan.HiddenSize || plan.EmbeddingLayout.TieWordEmbeddings != plan.ContextLayout.TieWordEmbeddings {
		t.Fatalf("embedding/context contract not applied: hidden=%d embedding=%+v context=%+v", plan.HiddenSize, plan.EmbeddingLayout, plan.ContextLayout)
	}
	if plan.RoPELayout.MaxPositionEmbeddings != plan.ContextLayout.MaxPositionEmbeddings || plan.RoPELayout.FullAttentionLayers != plan.FullAttentionLayers {
		t.Fatalf("rope/context contract not applied: rope=%+v context=%+v attention=%d", plan.RoPELayout, plan.ContextLayout, plan.FullAttentionLayers)
	}
	plan.EmbeddingLayout.VocabSize++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected embedding/context vocab mismatch")
	}
	plan.EmbeddingLayout.VocabSize = plan.ContextLayout.VocabSize
	plan.RoPELayout.MaxPositionEmbeddings++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected RoPE/context max position mismatch")
	}
}

func TestRuntimePlanAppliesMoELayoutContracts(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Experts != plan.RouterLayout.Experts || plan.ExpertsPerToken != plan.RouterLayout.ExpertsPerToken {
		t.Fatalf("router contract not applied: plan experts=%d active=%d layout=%+v", plan.Experts, plan.ExpertsPerToken, plan.RouterLayout)
	}
	if plan.MoEIntermediate != plan.FFNLayout.MoEIntermediate || plan.Experts != plan.FFNLayout.Experts || plan.ExpertsPerToken != plan.FFNLayout.ExpertsPerToken {
		t.Fatalf("FFN contract not applied: plan=%+v layout=%+v", plan, plan.FFNLayout)
	}
	plan.ExpertsPerToken++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected router/FFN active expert mismatch")
	}
	plan.ExpertsPerToken = plan.RouterLayout.ExpertsPerToken
	plan.MoEIntermediate++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected FFN intermediate mismatch")
	}
}
