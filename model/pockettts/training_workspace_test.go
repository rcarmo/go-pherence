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
	freshMetrics, fresh, err := PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	intoMetrics, into, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "workspace repeat loss", intoMetrics.Loss, freshMetrics.Loss, 0)
	freshMap := trainingStepGradientMap(fresh)
	for name, values := range trainingStepGradientMap(into) {
		assertSliceClose(t, "workspace repeat "+name, values, freshMap[name], 0)
	}
	if workspace.primaryFlowTape.fallbacks != 0 || workspace.endpointFlowTape.fallbacks != 0 || workspace.primaryFlowBackward.fallbacks != 0 || workspace.endpointFlowBackward.fallbacks != 0 {
		t.Fatalf("flow scratch spilled: primary=%d endpoint=%d backward=%d endpoint_backward=%d", workspace.primaryFlowTape.fallbacks, workspace.endpointFlowTape.fallbacks, workspace.primaryFlowBackward.fallbacks, workspace.endpointFlowBackward.fallbacks)
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
	flow.Time = append(flow.Time, flow.Time[0])
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted three-time flow head in existing workspace")
	}
	if _, err = NewTrainingWorkspace(lm, flow, batch); err == nil {
		t.Fatal("allocated a workspace for three-time flow head")
	}
	if _, _, err = PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("accepted three-time flow head in convenience step")
	}
	flow.Time = flow.Time[:2]
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
	flow = tinyTrainableFlowHead()
	flow.Condition = tinyLinear(4, 4, -.02)
	otherLM := tinyFlowLMTraining()
	if _, _, err = PocketTrainingStepInto(otherLM, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted workspace with a different FlowLM owner")
	}
	flow = tinyTrainableFlowHead()
	flow.Condition = tinyLinear(4, 4, -.02)
	workspace, err = NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	lm.Transformer.Layers = append(lm.Transformer.Layers, lm.Transformer.Layers[0])
	if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
		t.Fatal("accepted in-place FlowLM topology drift")
	}
	lm.Transformer.Layers = lm.Transformer.Layers[:len(lm.Transformer.Layers)-1]
	for name, drift := range map[string]func(){
		"width":          func() { lm.Transformer.Width++ },
		"heads":          func() { lm.Transformer.Heads++ },
		"head dimension": func() { lm.Transformer.HeadDim++ },
		"context":        func() { lm.Transformer.Context++ },
		"RoPE period":    func() { lm.Transformer.MaxPeriod++ },
	} {
		original := *lm.Transformer
		drift()
		if _, _, err = PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace); err == nil {
			t.Fatalf("accepted in-place FlowLM %s drift", name)
		}
		*lm.Transformer = original
	}
}
