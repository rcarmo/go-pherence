package pockettts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestFullTrainerPyTorchAdamWEMAAndResume(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	flowLM, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: append([]float32(nil), oracle.NormalizedLatents...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	_, gradients, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	config := AdamWConfig{LearningRate: oracle.AdamW.LearningRate, Beta1: oracle.AdamW.Beta1, Beta2: oracle.AdamW.Beta2, Epsilon: oracle.AdamW.Epsilon, WeightDecay: oracle.AdamW.WeightDecay}
	trainer, err := NewFullTrainer(flowLM, flow, weighting, config, oracle.AdamW.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	params, _ := fullParameterMap(flowLM, flow, weighting)
	if len(params) != len(oracle.AdamW.ParametersAfterStep) {
		t.Fatalf("updated params=%d want=%d", len(params), len(oracle.AdamW.ParametersAfterStep))
	}
	for name, values := range params {
		assertSliceClose(t, "full AdamW "+name, values, oracle.AdamW.ParametersAfterStep[name], 4e-6)
		assertSliceClose(t, "full EMA "+name, trainer.EMA[name], oracle.AdamW.EMAAfterStep[name], 4e-6)
	}
	trainer.FlowLM.LatentMean[0] = .125
	trainer.FlowLM.LatentStd[1] = 1.25
	trainer.Flow.Time[0].Frequencies[0] = .75
	state, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "nested", "full.json")
	if err = SaveFullTrainingState(path, state); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFullTrainingState(path)
	if err != nil {
		t.Fatal(err)
	}
	freshLM, freshFlow, freshWeight := trainingStepModelsFromOracle(t, oracle)
	resumed, err := NewFullTrainer(freshLM, freshFlow, freshWeight, config, oracle.AdamW.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	if err = resumed.LoadState(loaded); err != nil {
		t.Fatal(err)
	}
	resumedState, err := resumed.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state, resumedState) {
		t.Fatal("full state round-trip mismatch")
	}
	if resumed.FlowLM.LatentMean[0] != .125 || resumed.FlowLM.LatentStd[1] != 1.25 || resumed.Flow.Time[0].Frequencies[0] != .75 {
		t.Fatal("full buffers not restored")
	}
	_, gA, err := PocketTrainingStep(trainer.FlowLM, trainer.Flow, trainer.Weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	_, gB, err := PocketTrainingStep(resumed.FlowLM, resumed.Flow, resumed.Weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gA); err != nil {
		t.Fatal(err)
	}
	if err = resumed.Step(gB); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mustFullState(t, trainer), mustFullState(t, resumed)) {
		t.Fatal("full resumed second step differs")
	}
}

func loadTrainingStepOracle(t *testing.T) trainingStepOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle trainingStepOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	return oracle
}
func mustFullState(t *testing.T, trainer *FullTrainer) FullTrainingState {
	t.Helper()
	s, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	return s
}
