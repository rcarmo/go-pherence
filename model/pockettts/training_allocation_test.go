package pockettts

import "testing"

func TestPocketTrainingWarmAllocationBounds(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: append([]float32(nil), oracle.NormalizedLatents...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), TextTokens: append([]uint32(nil), oracle.TextTokens...)}
	samples := TrainingStepSamples{Mask: append([]bool(nil), oracle.Mask...), Noise: append([]float32(nil), oracle.Noise...), DiagonalTime: append([]float32(nil), oracle.DiagonalTime...), DistillS: append([]float32(nil), oracle.DistillS...), DistillT: append([]float32(nil), oracle.DistillT...)}
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	var gradients TrainingStepGradients
	stepAllocs := testing.AllocsPerRun(20, func() {
		_, gradients, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			panic(err)
		}
	})
	if stepAllocs > 720 {
		t.Fatalf("warm forward/backward allocs=%g, ceiling=720", stepAllocs)
	}
	trainer, err := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		t.Fatal(err)
	} // initialize moments outside measurement
	optimizerAllocs := testing.AllocsPerRun(100, func() {
		if err := trainer.Step(gradients); err != nil {
			panic(err)
		}
	})
	if optimizerAllocs != 0 {
		t.Fatalf("warm optimizer allocs=%g, want 0", optimizerAllocs)
	}
}
