package lfm2

import (
	"path/filepath"
	"testing"
)

func TestRoutingPlanFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := NewExecutionPlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRoutingPlan(meta.Config, exec)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Experts != 32 || plan.ExpertsPerToken != 4 || plan.DenseLayers != 2 || plan.MoELayers != 22 || plan.MoEIntermediate != 1792 {
		t.Fatalf("plan=%+v", plan)
	}
	if !plan.NormalizeTopK || !plan.UseExpertBias || plan.RoutedScalingFactor != 1.0 {
		t.Fatalf("routing flags=%+v", plan)
	}
}

func TestRoutingPlanRejectsMalformed(t *testing.T) {
	if err := (RoutingPlan{Experts: 4, ExpertsPerToken: 5, MoELayers: 1, MoEIntermediate: 16}).Validate(); err == nil {
		t.Fatal("expected active expert count error")
	}
	if err := (RoutingPlan{Experts: 4, ExpertsPerToken: 1, MoELayers: 0, MoEIntermediate: 16}).Validate(); err == nil {
		t.Fatal("expected moe layer count error")
	}
	if err := (RoutingPlan{Experts: 4, ExpertsPerToken: 1, MoELayers: 1, MoEIntermediate: 16, RoutedScalingFactor: -1}).Validate(); err == nil {
		t.Fatal("expected scaling factor error")
	}
}
