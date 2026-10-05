package pockettts

import (
	"testing"
)

func TestTrainingFlowTapeFullStepPyTorch(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	metrics, g, err := PocketTrainingStep(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "flow tape full loss", metrics.Loss, oracle.Metrics["loss"], 3e-5)
	got := trainingStepGradientMap(g)
	for name, values := range got {
		assertSliceClose(t, "flow tape full gradient "+name, values, oracle.Gradients[name], 5e-5)
	}
	assertSliceClose(t, "flow tape target gradients", g.Inputs.NormalizedLatents, oracle.DNormalizedLatents, 5e-5)
	assertSliceClose(t, "flow tape voice gradients", g.Inputs.VoiceLatents, oracle.DVoiceLatents, 5e-5)
}
