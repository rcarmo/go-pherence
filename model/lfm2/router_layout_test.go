package lfm2

import (
	"path/filepath"
	"testing"
)

func TestRouterLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := NewExecutionPlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewRouterLayout(meta.Config, exec)
	if err != nil {
		t.Fatal(err)
	}
	if layout.HiddenSize != 2048 || layout.Experts != 32 || layout.ExpertsPerToken != 4 || layout.MoELayers != 22 || !layout.UseExpertBias {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.RouterWeightFloats != 65536 || layout.RouterBiasFloats != 32 || layout.FloatsPerLayer != 65568 || layout.TotalRouterFloats != 1442496 {
		t.Fatalf("floats=%+v", layout)
	}
	scratch, err := layout.ScratchFloats(2)
	if err != nil {
		t.Fatal(err)
	}
	if scratch != 80 {
		t.Fatalf("scratch=%d", scratch)
	}
}

func TestRouterLayoutWithoutBias(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := meta.Config
	cfg.UseExpertBias = false
	layout, err := NewRouterLayout(cfg, ExecutionPlan{})
	if err != nil {
		t.Fatal(err)
	}
	if layout.UseExpertBias || layout.RouterBiasFloats != 0 || layout.FloatsPerLayer != 65536 {
		t.Fatalf("layout=%+v", layout)
	}
}

func TestRouterLayoutRejectsMalformed(t *testing.T) {
	bad := RouterLayout{HiddenSize: 2048, Experts: 32, ExpertsPerToken: 4, MoELayers: 22, UseExpertBias: true, RouterWeightFloats: 1, RouterBiasFloats: 32, FloatsPerLayer: 65568, TotalRouterFloats: 1442496, LogitsPerToken: 32, TopKPerToken: 4}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected router weight error")
	}
	bad = RouterLayout{HiddenSize: 2048, Experts: 32, ExpertsPerToken: 4, MoELayers: 22, UseExpertBias: true, RouterWeightFloats: 65536, RouterBiasFloats: 0, FloatsPerLayer: 65536, TotalRouterFloats: 1441792, LogitsPerToken: 32, TopKPerToken: 4}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected router bias error")
	}
	bad = RouterLayout{HiddenSize: 2048, Experts: 32, ExpertsPerToken: 4, MoELayers: 22, RouterWeightFloats: 65536, FloatsPerLayer: 65536, TotalRouterFloats: 1441792, LogitsPerToken: 1, TopKPerToken: 4}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected token output error")
	}
	if _, err := (RouterLayout{HiddenSize: 2048, Experts: 32, ExpertsPerToken: 4, MoELayers: 22, RouterWeightFloats: 65536, FloatsPerLayer: 65536, TotalRouterFloats: 1441792, LogitsPerToken: 32, TopKPerToken: 4}).ScratchFloats(-1); err == nil {
		t.Fatal("expected token count error")
	}
}
