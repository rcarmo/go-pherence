package hunyuan3d

import (
	"encoding/json"
	"fmt"
	"os"
)

const (
	ConditionerFixtureSchema = "go-pherence-hunyuan3d-conditioner-v1"
	DenoiserFixtureSchema    = "go-pherence-hunyuan3d-denoiser-step-v1"
	LowStepFixtureSchema     = "go-pherence-hunyuan3d-lowstep-latents-v1"
)

// TensorSummaryFixture is the shared compact-summary fixture envelope emitted
// by the conditioner, denoiser, and low-step latent Python fixture scripts.
type TensorSummaryFixture struct {
	Schema    string          `json:"schema"`
	Source    json.RawMessage `json:"source,omitempty"`
	LoadState json.RawMessage `json:"load_state,omitempty"`
	Scheduler json.RawMessage `json:"scheduler,omitempty"`
	Outputs   []TensorSummary `json:"outputs"`
	Steps     []struct {
		Index              int           `json:"index"`
		Timestep           float64       `json:"timestep"`
		ModelTimestepInput float64       `json:"model_timestep_input"`
		Latents            TensorSummary `json:"latents"`
	} `json:"steps,omitempty"`
}

func ReadTensorSummaryFixture(path string, allowedSchemas ...string) (TensorSummaryFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return TensorSummaryFixture{}, err
	}
	var fixture TensorSummaryFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return TensorSummaryFixture{}, fmt.Errorf("parse Hunyuan3D tensor summary fixture %s: %w", path, err)
	}
	if !schemaAllowed(fixture.Schema, allowedSchemas) {
		return TensorSummaryFixture{}, fmt.Errorf("unsupported Hunyuan3D fixture schema %q", fixture.Schema)
	}
	if len(fixture.Outputs) == 0 {
		return TensorSummaryFixture{}, fmt.Errorf("Hunyuan3D fixture %q has no outputs", fixture.Schema)
	}
	return fixture, nil
}

func ValidateRequiredOutputs(fixture TensorSummaryFixture, names ...string) error {
	for _, name := range names {
		if _, ok := FindTensorSummary(fixture.Outputs, name); !ok {
			return fmt.Errorf("Hunyuan3D fixture %q missing output %q", fixture.Schema, name)
		}
	}
	return nil
}

func CompareRequiredOutputs(got []TensorSummary, fixture TensorSummaryFixture, tolerance float32, names ...string) error {
	if err := ValidateRequiredOutputs(fixture, names...); err != nil {
		return err
	}
	for _, name := range names {
		g, ok := FindTensorSummary(got, name)
		if !ok {
			return fmt.Errorf("missing Go tensor summary %q", name)
		}
		w, _ := FindTensorSummary(fixture.Outputs, name)
		if err := CompareTensorSummary(g, w, tolerance); err != nil {
			return err
		}
	}
	return nil
}

func schemaAllowed(schema string, allowed []string) bool {
	if len(allowed) == 0 {
		return schema != ""
	}
	for _, want := range allowed {
		if schema == want {
			return true
		}
	}
	return false
}
