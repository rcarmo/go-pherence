package pockettts

import "testing"

func TestTrainingWorkspaceParityReuseAndAliasing(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: append([]float32(nil), oracle.NormalizedLatents...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	ownedMetrics, owned, err := PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	intoMetrics, into, err := PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "workspace loss", intoMetrics.Loss, ownedMetrics.Loss, 0)
	for name, values := range trainingStepGradientMap(into) {
		assertSliceClose(t, "workspace "+name, values, trainingStepGradientMap(owned)[name], 0)
	}
	pointer := &into.Flow.Input.Weight[0]
	before := *pointer
	samples.DiagonalTime[0] += .01
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err != nil {
		t.Fatal(err)
	}
	if *pointer == before {
		t.Fatal("workspace result did not update in place")
	}
}
func TestTrainingWorkspaceRejectsShape(t *testing.T) {
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	batch, samples := tinyFlowLMBatch(), tinyTrainingStepSamples()
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	batch.Frames = 2
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted workspace shape mismatch")
	}
	batch = tinyFlowLMBatch()
	flow.Blocks = append(flow.Blocks, flow.Blocks[0])
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted workspace flow topology mismatch")
	}
	flow = tinyTrainableFlowHead()
	flow.Condition = tinyLinear(4, 4, -.02)
	workspace, err = NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	flow.Input.Bias = nil
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted workspace bias topology mismatch")
	}
}
