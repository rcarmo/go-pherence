package hunyuan3d

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadTensorSummaryFixtureAndRequiredOutputs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.json")
	data := `{
  "schema": "go-pherence-hunyuan3d-denoiser-step-v1",
  "source": {"seed": 1234},
  "load_state": {"model_tensor_count": 2},
  "outputs": [
    {"name":"latents","dtype":"float32","shape":[1,512,64],"sha256_le_f32":"a","min":-1,"max":1,"mean":0},
    {"name":"denoiser_output","dtype":"float32","shape":[1,512,64],"sha256_le_f32":"b","min":-2,"max":2,"mean":0.5}
  ]
}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	fixture, err := ReadTensorSummaryFixture(path, DenoiserFixtureSchema)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateRequiredOutputs(fixture, "latents", "denoiser_output"); err != nil {
		t.Fatal(err)
	}
	if err := CompareRequiredOutputs(fixture.Outputs, fixture, 1e-6, "latents", "denoiser_output"); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTensorSummaryFixture(path, ConditionerFixtureSchema); err == nil {
		t.Fatal("wrong schema accepted")
	}
	if err := ValidateRequiredOutputs(fixture, "missing"); err == nil {
		t.Fatal("missing required output accepted")
	}
}

func TestReadTensorSummaryFixtureLowStepSteps(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lowstep.json")
	data := `{
  "schema": "go-pherence-hunyuan3d-lowstep-latents-v1",
  "scheduler": {"num_train_timesteps": 1000},
  "outputs": [
    {"name":"initial_latents","dtype":"float32","shape":[1,512,64],"sha256_le_f32":"a","min":0,"max":0,"mean":0},
    {"name":"final_latents","dtype":"float32","shape":[1,512,64],"sha256_le_f32":"b","min":1,"max":1,"mean":1}
  ],
  "steps": [
    {"index":0,"timestep":0,"model_timestep_input":0,"latents":{"name":"latents_after_step_0","dtype":"float32","shape":[1,512,64],"sha256_le_f32":"c","min":0,"max":1,"mean":0.5}}
  ]
}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	fixture, err := ReadTensorSummaryFixture(path, LowStepFixtureSchema)
	if err != nil {
		t.Fatal(err)
	}
	if len(fixture.Steps) != 1 || fixture.Steps[0].Latents.Name != "latents_after_step_0" {
		t.Fatalf("steps=%+v", fixture.Steps)
	}
	if err := ValidateRequiredOutputs(fixture, "initial_latents", "final_latents"); err != nil {
		t.Fatal(err)
	}
}
