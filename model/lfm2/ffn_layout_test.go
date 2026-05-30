package lfm2

import (
	"path/filepath"
	"testing"
)

func TestFFNLayoutFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := NewExecutionPlan(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewFFNLayout(meta.Config, exec)
	if err != nil {
		t.Fatal(err)
	}
	if layout.HiddenSize != 2048 || layout.DenseIntermediate != 7168 || layout.MoEIntermediate != 1792 || layout.DenseLayers != 2 || layout.MoELayers != 22 {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.DenseParamsPerLayer != 44040192 || layout.ExpertParamsPerExpert != 11010048 {
		t.Fatalf("params dense=%d expert=%d", layout.DenseParamsPerLayer, layout.ExpertParamsPerExpert)
	}
}

func TestFFNLayoutRejectsMalformed(t *testing.T) {
	bad := FFNLayout{HiddenSize: 2048, DenseIntermediate: 7168, MoEIntermediate: 1792, DenseLayers: 2, MoELayers: 22, Experts: 32, ExpertsPerToken: 4, DenseParamsPerLayer: 1, ExpertParamsPerExpert: 11010048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected dense param mismatch")
	}
	bad = FFNLayout{HiddenSize: 2048, DenseIntermediate: 7168, MoEIntermediate: 1792, DenseLayers: 2, MoELayers: 22, Experts: 32, ExpertsPerToken: 33, DenseParamsPerLayer: 44040192, ExpertParamsPerExpert: 11010048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected expert count error")
	}
}
