package pockettts

import "testing"

func TestFullTrainerRejectsTopologyRebind(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		t.Fatal(err)
	}
	replacement, _, _ := trainingStepModelsFromOracle(t, oracle)
	trainer.FlowLM = replacement
	if _, err = trainer.State(); err == nil {
		t.Fatal("accepted full model pointer rebind")
	}
	trainer.FlowLM = lm
	flow.Blocks = append(flow.Blocks, flow.Blocks[0])
	if _, err = trainer.State(); err == nil {
		t.Fatal("accepted expanded flow topology")
	}
}

func TestFullTrainerFollowsSameShapeParameterRebind(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	_, grads, err := PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		t.Fatal(err)
	}
	old := lm.BOS
	lm.BOS = append([]float32(nil), old...)
	before := lm.BOS[0]
	if err = trainer.Step(grads); err != nil {
		t.Fatal(err)
	}
	if lm.BOS[0] == before {
		t.Fatal("rebound live parameter was not updated")
	}
	if old[0] != before {
		t.Fatal("stale rebound parameter was updated")
	}
	state, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, p := range state.Params {
		if p.Name == "flow_lm.bos_emb" {
			found = true
			assertSliceClose(t, "rebound state", p.Values, lm.BOS, 0)
		}
	}
	if !found {
		t.Fatal("rebound state missing BOS")
	}
}
