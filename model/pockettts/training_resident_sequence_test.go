//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"reflect"
	"testing"
)

func TestTrainingResidentSequenceRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-sequence-hybrid-v1", "vulkan-resident-sequence-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			a, b, d := residentGraphRunner(t, scope)
			if _, err := a.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "sequence", limits); err != nil {
				t.Fatal(err)
			}
			resumed, rb, _ := residentGraphRunner(t, scope)
			if err := resumed.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			if rb.state != nil {
				t.Fatal("stale state")
			}
			allocations := d.allocations
			for _, r := range []*TrainingRunner{a, resumed} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for name, p := range a.trainer.params {
				assertSliceClose(t, "sequence resume "+name, resumed.trainer.params[name], p, 1e-6)
				assertSliceClose(t, "sequence resume M "+name, resumed.trainer.Moments[name].M, a.trainer.Moments[name].M, 1e-6)
				assertSliceClose(t, "sequence resume V "+name, resumed.trainer.Moments[name].V, a.trainer.Moments[name].V, 1e-6)
				assertSliceClose(t, "sequence resume EMA "+name, resumed.trainer.EMA[name], a.trainer.EMA[name], 1e-6)
			}
			if allocations != d.allocations || !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, resumed.run)) {
				t.Fatal("resident allocations/replay")
			}
			legacy, _, _ := residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
			before := mustFullState(t, legacy.trainer)
			if err := legacy.Restore(dir, limits); err == nil {
				t.Fatal("cross-scope restore")
			}
			if !reflect.DeepEqual(before, mustFullState(t, legacy.trainer)) {
				t.Fatal("restore mutated")
			}
			if b.Close() != nil || rb.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentSequenceEmptyAndRepeatedParity(t *testing.T) {
	for _, mode := range []string{"empty-voice", "repeated-tokens", "one-frame"} {
		t.Run(mode, func(t *testing.T) {
			oracle := loadTrainingStepOracle(t)
			lm, flow, w := trainingStepModelsFromOracle(t, oracle)
			cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
			trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
			if err != nil {
				t.Fatal(err)
			}
			backend, d := installResidentGraph(t, trainer, "vulkan-resident-sequence-hybrid-v1")
			batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
			samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
			switch mode {
			case "empty-voice":
				batch.VoiceFrames = 0
				batch.VoiceLatents = nil
			case "repeated-tokens":
				batch.TextTokens = []uint32{oracle.TextTokens[0], oracle.TextTokens[0], oracle.TextTokens[0]}
			case "one-frame":
				batch.Frames = 1
				batch.NormalizedLatents = batch.NormalizedLatents[:lm.LatentDim]
				samples.Mask = []bool{true}
				samples.Noise = samples.Noise[:lm.LatentDim]
				samples.DiagonalTime = samples.DiagonalTime[:1]
				samples.DistillS = samples.DistillS[:1]
				samples.DistillT = samples.DistillT[:1]
			}
			cfg := DefaultTrainingStepConfig()
			want, wg, err := PocketTrainingStep(cpuLM, cpuFlow, cpuW, batch, samples, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := backend.PrepareTrainingStep(context.Background(), trainer); err != nil {
				t.Fatal(err)
			}
			got, gg, err := PocketTrainingStep(lm, flow, w, batch, samples, cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertClose64(t, "sequence variant loss", got.Loss, want.Loss, 3e-5)
			assertSliceClose(t, "sequence variant audio input", gg.Inputs.NormalizedLatents, wg.Inputs.NormalizedLatents, 5e-5)
			assertSliceClose(t, "sequence variant voice input", gg.Inputs.VoiceLatents, wg.Inputs.VoiceLatents, 5e-5)
			if err := backend.UpdateTrainer(context.Background(), trainer, gg, 1e20); err != nil {
				t.Fatal(err)
			}
			wantGradients := trainingStepGradientMap(wg)
			for i, name := range trainer.names {
				assertSliceClose(t, "sequence variant gradient "+name, d.validatedGradients[i], wantGradients[name], 5e-5)
			}
		})
	}
}
func TestTrainingResidentSequenceFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-sequence-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-sequence-hybrid-v1")
		d.failTape = at
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("failure admitted", at)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed step mutated", at)
		}
		calls := d.tapes
		if _, err := r.Step(context.Background()); err == nil || calls != d.tapes {
			t.Fatal("retry", at)
		}
		if b.Close() == nil {
			t.Fatal("owner released", at)
		}
		if err := b.ValidateBoundary(r.trainer); err == nil {
			t.Fatal("failed boundary", at)
		}
	}
	r, b, _ := residentGraphRunner(t, "ptx-resident-sequence-hybrid-v1")
	r.trainer.FlowLM.SequenceBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("removed owner")
	}
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil {
		t.Fatal("admission owner")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
	legacy.trainer.FlowLM.SequenceBackend = lb
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("legacy injection")
	}
}
func TestTrainingResidentSequenceAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-sequence-hybrid-v1")
	batch := r.rows[0].Batch
	if _, err := b.EvaluateSequence(r.trainer.FlowLM, batch, nil, nil, nil, false); err == nil {
		t.Fatal("inactive")
	}
	if _, err := b.EvaluateAudioAssembly(r.trainer.FlowLM, batch, nil, false); err == nil {
		t.Fatal("inactive audio")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateSequence(tinyFlowLMTraining(), batch, nil, nil, nil, false); err == nil {
		t.Fatal("wrong model")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-sequence-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateAudioAssembly(tinyFlowLMTraining(), batch, nil, false); err == nil {
		t.Fatal("wrong audio model")
	}
}
