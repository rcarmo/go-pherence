package pockettts

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// This opt-in gate checks released-shape in-memory resume. An explicit
// GO_PHERENCE_POCKETTTS_RELEASED_BINARY_DIR additionally exercises binary
// checkpoint publication and reload. It does not qualify synthetic latents
// as training data or exercise the legacy JSON checkpoint path.
func TestReleasedTrainingInMemoryResume(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_RELEASED_RESUME") != "1" {
		t.Skip("set GO_PHERENCE_POCKETTTS_RELEASED_RESUME=1")
	}
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(t, 32)
	workspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	continuous, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	step := func(trainer *FullTrainer, workspace *TrainingWorkspace) {
		t.Helper()
		metrics, gradients, err := PocketTrainingStepInto(trainer.FlowLM, trainer.Flow, trainer.Weighting, batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil || !isFinite(float32(metrics.Loss)) {
			t.Fatalf("released training forward/backward: metrics=%+v err=%v", metrics, err)
		}
		if err = trainer.Step(gradients); err != nil {
			t.Fatal(err)
		}
	}
	step(continuous, workspace)
	state, err := continuous.State()
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := ""
	if dir := os.Getenv("GO_PHERENCE_POCKETTTS_RELEASED_BINARY_DIR"); dir != "" {
		if info, statErr := os.Stat(dir); statErr != nil || !info.IsDir() {
			t.Fatalf("released checkpoint directory %q must exist: %v", dir, statErr)
		}
		tempDir, tempErr := os.MkdirTemp(dir, "pockettts-released-resume-*")
		if tempErr != nil {
			t.Fatal(tempErr)
		}
		defer os.RemoveAll(tempDir)
		checkpoint = filepath.Join(tempDir, "training.bin")
		if err = SaveFullTrainingStateBinary(checkpoint, state); err != nil {
			t.Fatal(err)
		}
		info, statErr := os.Stat(checkpoint)
		if statErr != nil {
			t.Fatal(statErr)
		}
		t.Logf("released binary checkpoint bytes=%d", info.Size())
		loaded, loadErr := LoadFullTrainingStateBinary(checkpoint, 2<<30)
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		if !reflect.DeepEqual(state, loaded) {
			t.Fatal("released binary checkpoint differs from in-memory state")
		}
		state = loaded
	}
	freshLM, freshFlow, freshWeight, _ := releasedTrainingModels(t)
	resumed, err := NewFullTrainer(freshLM, freshFlow, freshWeight, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	if err = resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	// Check every parameter, mutable buffer, Adam moment and EMA value, not just
	// a sampled layer or the step counter.
	got, err := resumed.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state, got) {
		t.Fatal("released training state differs immediately after in-memory resume")
	}
	state, got = FullTrainingState{}, FullTrainingState{}
	resumedWorkspace, err := NewAdmittedTrainingWorkspace(freshLM, freshFlow, freshWeight, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	step(continuous, workspace)
	step(resumed, resumedWorkspace)
	want, err := continuous.State()
	if err != nil {
		t.Fatal(err)
	}
	got, err = resumed.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatal("released training state diverged after resumed update")
	}
	if continuous.StepCount != 2 || resumed.StepCount != 2 {
		t.Fatalf("released training step counters: continuous=%d resumed=%d", continuous.StepCount, resumed.StepCount)
	}
	if checkpoint == "" {
		return
	}
	// Replace the same released-size checkpoint after a resumed update, then
	// resume again into fresh models and compare the next complete update.
	if err = SaveFullTrainingStateBinary(checkpoint, got); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFullTrainingStateBinary(checkpoint, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, loaded) {
		t.Fatal("second released binary checkpoint differs from continuous state")
	}
	freshLM, freshFlow, freshWeight, _ = releasedTrainingModels(t)
	resumed, err = NewFullTrainer(freshLM, freshFlow, freshWeight, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	if err = resumed.LoadState(loaded); err != nil {
		t.Fatal(err)
	}
	restored, stateErr := resumed.State()
	if stateErr != nil {
		t.Fatal(stateErr)
	}
	if !reflect.DeepEqual(loaded, restored) {
		t.Fatal("released training state differs after second binary resume")
	}
	want, got, loaded, restored = FullTrainingState{}, FullTrainingState{}, FullTrainingState{}, FullTrainingState{}
	resumedWorkspace, err = NewAdmittedTrainingWorkspace(freshLM, freshFlow, freshWeight, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	step(continuous, workspace)
	step(resumed, resumedWorkspace)
	want, err = continuous.State()
	if err != nil {
		t.Fatal(err)
	}
	got, err = resumed.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) || continuous.StepCount != 3 || resumed.StepCount != 3 {
		t.Fatal("released training state diverged after second binary resume")
	}
}
