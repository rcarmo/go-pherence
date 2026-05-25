package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Trellis2LowStepFixture captures compact summaries produced by
// scripts/trellis2_lowstep_fixture.py. It intentionally stores hashes and shape
// metadata rather than full tensors.
type Trellis2LowStepFixture struct {
	Schema          string                       `json:"schema"`
	ModelDir        string                       `json:"model_dir,omitempty"`
	ConfigFile      string                       `json:"config_file,omitempty"`
	Device          string                       `json:"device,omitempty"`
	Seed            int64                        `json:"seed,omitempty"`
	Steps           int                          `json:"steps,omitempty"`
	CondResolution  int                          `json:"cond_resolution,omitempty"`
	Summaries       []FixtureTensorSummary       `json:"summaries,omitempty"`
	SparseStructure *Trellis2SparseCoordSummary  `json:"sparse_structure,omitempty"`
	ShapeSLat       *Trellis2SparseLatentSummary `json:"shape_slat,omitempty"`
}

// Trellis2SparseCoordSummary stores sparse structure coordinate metadata. The
// Python fixture writes int32 little-endian hashes for coords.
type Trellis2SparseCoordSummary struct {
	Shape       []int  `json:"shape"`
	DType       string `json:"dtype"`
	SHA256LEI32 string `json:"sha256_le_i32"`
	FirstValues []int  `json:"first_values,omitempty"`
}

type Trellis2SparseLatentSummary struct {
	CoordsShape []int  `json:"coords_shape,omitempty"`
	FlowModel   string `json:"flow_model,omitempty"`
}

func ReadTrellis2LowStepFixture(path string) (*Trellis2LowStepFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var fixture Trellis2LowStepFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return nil, err
	}
	if err := ValidateTrellis2LowStepFixture(&fixture); err != nil {
		return nil, err
	}
	return &fixture, nil
}

func ValidateTrellis2LowStepFixture(f *Trellis2LowStepFixture) error {
	if f == nil {
		return fmt.Errorf("trellis2 fixture: nil")
	}
	if f.Schema != "go-pherence-trellis2-lowstep-v1" {
		return fmt.Errorf("trellis2 fixture: unsupported schema %q", f.Schema)
	}
	if f.Steps <= 0 {
		return fmt.Errorf("trellis2 fixture: invalid steps %d", f.Steps)
	}
	if f.CondResolution <= 0 {
		return fmt.Errorf("trellis2 fixture: invalid cond resolution %d", f.CondResolution)
	}
	for i, s := range f.Summaries {
		if err := ValidateFixtureTensorSummary(s); err != nil {
			return fmt.Errorf("trellis2 fixture summary %d: %w", i, err)
		}
	}
	if f.SparseStructure != nil {
		if err := ValidateTrellis2SparseCoordSummary(*f.SparseStructure); err != nil {
			return err
		}
	}
	return nil
}

func ValidateFixtureTensorSummary(s FixtureTensorSummary) error {
	if s.Name == "" {
		return fmt.Errorf("fixture tensor summary: missing name")
	}
	if s.DType != "float32" {
		return fmt.Errorf("fixture tensor summary %q: unsupported dtype %q", s.Name, s.DType)
	}
	if _, err := FixtureShapeNumel(s.Shape); err != nil {
		return fmt.Errorf("fixture tensor summary %q: %w", s.Name, err)
	}
	if s.SHA256LEF32 == "" {
		return fmt.Errorf("fixture tensor summary %q: missing sha256_le_f32", s.Name)
	}
	return nil
}

func ValidateTrellis2SparseCoordSummary(s Trellis2SparseCoordSummary) error {
	if len(s.Shape) != 2 || s.Shape[1] != 4 {
		return fmt.Errorf("trellis2 sparse coords: expected [N,4] shape, got %v", s.Shape)
	}
	if s.DType != "int32" && s.DType != "int" {
		return fmt.Errorf("trellis2 sparse coords: unsupported dtype %q", s.DType)
	}
	if s.SHA256LEI32 == "" {
		return fmt.Errorf("trellis2 sparse coords: missing sha256_le_i32")
	}
	return nil
}
