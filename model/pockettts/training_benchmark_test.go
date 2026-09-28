package pockettts

import (
	"encoding/json"
	"os"
	"testing"
)

func loadBenchmarkTrainingFixture(b *testing.B) (*FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, FlowLMTrainingBatch, TrainingStepSamples, TrainingStepConfig) {
	b.Helper()
	data, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		b.Fatal(err)
	}
	var oracle trainingStepOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		b.Fatal(err)
	}
	lm, flow, w := trainingStepModelsFromOracle(b, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: append([]float32(nil), oracle.NormalizedLatents...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), TextTokens: append([]uint32(nil), oracle.TextTokens...)}
	samples := TrainingStepSamples{Mask: append([]bool(nil), oracle.Mask...), Noise: append([]float32(nil), oracle.Noise...), DiagonalTime: append([]float32(nil), oracle.DiagonalTime...), DistillS: append([]float32(nil), oracle.DistillS...), DistillT: append([]float32(nil), oracle.DistillT...)}
	return lm, flow, w, batch, samples, DefaultTrainingStepConfig()
}

func BenchmarkPocketTrainingWorkspaceSetup(b *testing.B) {
	lm, flow, _, batch, _, _ := loadBenchmarkTrainingFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := NewTrainingWorkspace(lm, flow, batch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPocketTrainingTrainerSetup(b *testing.B) {
	lm, flow, w, _, _, _ := loadBenchmarkTrainingFixture(b)
	config := AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := NewFullTrainer(lm, flow, w, config, .9); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPocketTrainingColdSetup(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _, _ = loadBenchmarkTrainingFixture(b)
	}
}
func BenchmarkPocketTrainingForwardBackward(b *testing.B) {
	lm, flow, w, batch, samples, cfg := loadBenchmarkTrainingFixture(b)
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := PocketTrainingStepInto(lm, flow, w, batch, samples, cfg, workspace); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkPocketTrainingOptimizerEMA(b *testing.B) {
	lm, flow, w, batch, samples, cfg := loadBenchmarkTrainingFixture(b)
	_, grads, err := PocketTrainingStep(lm, flow, w, batch, samples, cfg)
	if err != nil {
		b.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		b.Fatal(err)
	}
	if err := trainer.Step(grads); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := trainer.Step(grads); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkPocketTrainingFullStep(b *testing.B) {
	lm, flow, w, batch, samples, cfg := loadBenchmarkTrainingFixture(b)
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		b.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		b.Fatal(err)
	}
	_, warmGradients, err := PocketTrainingStepInto(lm, flow, w, batch, samples, cfg, workspace)
	if err != nil {
		b.Fatal(err)
	}
	if err = trainer.Step(warmGradients); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_, grads, err := PocketTrainingStepInto(lm, flow, w, batch, samples, cfg, workspace)
		if err != nil {
			b.Fatal(err)
		}
		if err = trainer.Step(grads); err != nil {
			b.Fatal(err)
		}
	}
}
