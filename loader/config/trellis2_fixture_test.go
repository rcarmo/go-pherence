package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadTrellis2LowStepFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.json")
	json := `{
		"schema":"go-pherence-trellis2-lowstep-v1",
		"model_dir":"microsoft/TRELLIS.2-4B",
		"config_file":"pipeline.json",
		"device":"cuda",
		"seed":1234,
		"steps":2,
		"cond_resolution":512,
		"summaries":[{
			"name":"cond.cond",
			"dtype":"float32",
			"shape":[1,1024],
			"sha256_le_f32":"0123456789abcdef",
			"min":-1.0,
			"max":1.0,
			"mean":0.0,
			"first_values":[0.0,1.0]
		}],
		"sparse_structure":{
			"shape":[8,4],
			"dtype":"int32",
			"sha256_le_i32":"abcdef0123456789",
			"first_values":[0,1,2,3]
		},
		"shape_slat":{
			"coords_shape":[8,4],
			"flow_model":"shape_slat_flow_model_512"
		}
	}`
	if err := os.WriteFile(path, []byte(json), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTrellis2LowStepFixture(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != "go-pherence-trellis2-lowstep-v1" || got.Steps != 2 || got.SparseStructure.Shape[1] != 4 {
		t.Fatalf("fixture=%+v", got)
	}
	if len(got.Summaries) != 1 || got.Summaries[0].Name != "cond.cond" {
		t.Fatalf("summaries=%+v", got.Summaries)
	}
}

func TestValidateTrellis2LowStepFixtureRejectsBadShape(t *testing.T) {
	fixture := &Trellis2LowStepFixture{
		Schema:         "go-pherence-trellis2-lowstep-v1",
		Steps:          1,
		CondResolution: 512,
		SparseStructure: &Trellis2SparseCoordSummary{
			Shape:       []int{8, 3},
			DType:       "int32",
			SHA256LEI32: "abc",
		},
	}
	if err := ValidateTrellis2LowStepFixture(fixture); err == nil {
		t.Fatal("bad sparse coord shape accepted")
	}
}

func TestValidateFixtureTensorSummaryRejectsBadSummary(t *testing.T) {
	if err := ValidateFixtureTensorSummary(FixtureTensorSummary{Name: "x", DType: "float16", Shape: []int{1}, SHA256LEF32: "abc"}); err == nil {
		t.Fatal("bad dtype accepted")
	}
	if err := ValidateFixtureTensorSummary(FixtureTensorSummary{Name: "x", DType: "float32", Shape: []int{0}, SHA256LEF32: "abc"}); err == nil {
		t.Fatal("bad shape accepted")
	}
	if err := ValidateFixtureTensorSummary(FixtureTensorSummary{Name: "x", DType: "float32", Shape: []int{1}}); err == nil {
		t.Fatal("missing hash accepted")
	}
}
