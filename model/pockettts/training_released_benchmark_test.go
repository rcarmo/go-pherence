package pockettts

import "testing"

func releasedTrainingRow(tb testing.TB, textTokens int) (*FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, FlowLMTrainingBatch, TrainingStepSamples, TrainingShapePlan) {
	tb.Helper()
	lm, flow, weighting, cfg := releasedTrainingModels(tb)
	frames, voice := ProductionTrainingTargetFrames, ProductionTrainingVoiceFrames
	batch := FlowLMTrainingBatch{
		Frames: frames, VoiceFrames: voice,
		NormalizedLatents: make([]float32, frames*lm.LatentDim),
		VoiceLatents:      make([]float32, voice*lm.LatentDim),
		TextTokens:        make([]uint32, textTokens),
	}
	samples := TrainingStepSamples{
		Mask:         make([]bool, frames),
		Noise:        make([]float32, frames*lm.LatentDim),
		DiagonalTime: make([]float32, frames),
		DistillS:     make([]float32, frames),
		DistillT:     make([]float32, frames),
	}
	for row := 0; row < frames; row++ {
		samples.Mask[row] = true
		samples.DiagonalTime[row] = .2 + .6*float32(row%17)/16
		samples.DistillS[row] = .1 + .3*float32(row%13)/12
		samples.DistillT[row] = samples.DistillS[row] + .5
		for channel := 0; channel < lm.LatentDim; channel++ {
			i := row*lm.LatentDim + channel
			batch.NormalizedLatents[i] = float32((i%29)-14) / 29
			samples.Noise[i] = float32((i%31)-15) / 31
		}
	}
	for i := range batch.VoiceLatents {
		batch.VoiceLatents[i] = float32((i%23)-11) / 23
	}
	for i := range batch.TextTokens {
		batch.TextTokens[i] = uint32(1 + i%(lm.Vocabulary-1))
	}
	shape := TrainingShape{MicroBatchRows: 1, TargetFrames: frames, VoiceFrames: voice, TextTokens: textTokens, GradientAccumulation: 1, FlowBatchMultiplier: 1}
	plan, err := PlanTrainingShape(cfg, shape, ProductionTrainingShapeLimits(64<<30))
	if err != nil {
		tb.Fatal(err)
	}
	return lm, flow, weighting, batch, samples, plan
}

func BenchmarkReleasedPocketTrainingProductionRowWorkspace(b *testing.B) {
	lm, flow, weighting, batch, _, plan := releasedTrainingRow(b, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReleasedPocketTrainingProductionRowForwardBackward(b *testing.B) {
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(b, 32)
	workspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		metrics, gradients, err := PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			b.Fatal(err)
		}
		if !isFinite(float32(metrics.Loss)) || gradients.FlowLM == nil || gradients.Flow == nil || gradients.Weighting == nil {
			b.Fatal("invalid released production-row result")
		}
	}
}

func BenchmarkReleasedPocketTrainingProductionRowWarmForwardBackward(b *testing.B) {
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(b, 32)
	workspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		b.Fatal(err)
	}
	if _, _, err = PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		metrics, gradients, err := PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			b.Fatal(err)
		}
		if !isFinite(float32(metrics.Loss)) || gradients.FlowLM == nil || gradients.Flow == nil || gradients.Weighting == nil || workspace.primaryFlowTape.fallbacks != 0 || workspace.endpointFlowTape.fallbacks != 0 || workspace.primaryFlowBackward.fallbacks != 0 || workspace.endpointFlowBackward.fallbacks != 0 {
			b.Fatal("invalid warm released production-row result or flow tape scratch spill")
		}
	}
}

func BenchmarkReleasedPocketTrainingProductionRowOptimizerEMA(b *testing.B) {
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(b, 32)
	workspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		b.Fatal(err)
	}
	_, gradients, err := PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), workspace)
	if err != nil {
		b.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		b.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err = trainer.Step(gradients); err != nil {
			b.Fatal(err)
		}
	}
}
