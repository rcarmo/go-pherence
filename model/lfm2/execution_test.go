package lfm2

import (
	"path/filepath"
	"testing"
)

func TestExecutionPlanFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewExecutionPlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Steps) != 24 || len(plan.DenseIndices) != 2 || len(plan.MoEIndices) != 22 {
		t.Fatalf("plan=%+v", plan)
	}
	for _, idx := range []int{0, 1} {
		if !plan.IsDenseLayer(idx) || plan.IsMoELayer(idx) {
			t.Fatalf("dense membership failed for %d: %+v", idx, plan)
		}
	}
	for _, idx := range []int{2, 7, 23} {
		if !plan.IsMoELayer(idx) || plan.IsDenseLayer(idx) {
			t.Fatalf("moe membership failed for %d: %+v", idx, plan)
		}
	}
	if plan.Steps[7].Kind != LayerFullAttention || plan.Steps[7].FFN != FFNMoE {
		t.Fatalf("layer 7=%+v", plan.Steps[7])
	}
}

func TestExecutionPlanRejectsMalformed(t *testing.T) {
	p := ExecutionPlan{Steps: []LayerExecutionStep{{Index: 1, Kind: LayerConv, FFN: FFNDense}}, DenseIndices: []int{1}}
	if err := p.Validate(1); err == nil {
		t.Fatal("expected non-sequential index error")
	}
	p = ExecutionPlan{Steps: []LayerExecutionStep{{Index: 0, Kind: LayerConv, FFN: "bad"}}}
	if err := p.Validate(1); err == nil {
		t.Fatal("expected bad ffn kind error")
	}
}
