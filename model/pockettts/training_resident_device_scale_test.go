//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

func TestTrainingResidentDeviceScaleRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-devicescale-hybrid-v1", "vulkan-resident-devicescale-hybrid-v1"} {
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
			legacy, _, _ := residentGraphRunner(t, "ptx-resident-pruned-hybrid-v1")
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
func TestTrainingResidentDeviceScaleFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
	r.trainer.Flow.StepReductionBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("removed owner")
	}
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil {
		t.Fatal("admission owner")
	}
	legacy, _, _ := residentGraphRunner(t, "ptx-resident-pruned-hybrid-v1")
	legacy.trainer.Flow.TrainingBackend = b
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("legacy injection")
	}
}
func TestTrainingResidentDeviceScaleObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-devicescale-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentDeviceScaleAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
	in := stepReductionFixture(3, 2)
	if _, err := b.EvaluateStepReduction(r.trainer.Flow, in, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("inactive")
	}
	if _, err := b.EvaluateStepInputGradient(r.trainer.Flow, []float32{1}, []float32{2}); err == nil {
		t.Fatal("inactive input")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateStepInputGradient(r.trainer.Flow, []float32{1}, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateStepReduction(&FlowHeadCPU{}, in, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("owner")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	b.MaxTapeBytes = 1
	if _, err := b.EvaluateStepReduction(r.trainer.Flow, in, DefaultTrainingStepConfig()); err == nil || b.terminal == nil {
		t.Fatal("budget")
	}
	legacy, _, _ := residentGraphRunner(t, "ptx-resident-pruned-hybrid-v1")
	legacy.trainer.Flow.StepReductionBackend = b
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("legacy injection")
	}
}
func TestTrainingResidentDeviceScaleInputAdmission(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "budget"} {
		r, b, _ := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		m := r.trainer.Flow
		a, c := []float32{1}, []float32{2}
		switch kind {
		case "owner":
			m = &FlowHeadCPU{}
		case "shape":
			c = nil
		case "budget":
			b.MaxTapeBytes = 1
		}
		if _, err := b.EvaluateStepInputGradient(m, a, c); err == nil || b.terminal == nil {
			t.Fatal("step input admitted", kind)
		}
	}
}
func TestTrainingResidentDeviceScalePaddingParity(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-devicescale-hybrid-v1")
	mask := make([]bool, oracle.Frames)
	mask[0] = true
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	cfg := DefaultTrainingStepConfig()
	want, wg, err := PocketTrainingStep(cpuLM, cpuFlow, cpuW, batch, samples, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	got, g, err := PocketTrainingStep(lm, flow, w, batch, samples, cfg)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "step padding loss", got.Loss, want.Loss, 3e-5)
	assertSliceClose(t, "step padding target", g.Inputs.NormalizedLatents, wg.Inputs.NormalizedLatents, 5e-5)
	assertSliceClose(t, "step padding voice", g.Inputs.VoiceLatents, wg.Inputs.VoiceLatents, 5e-5)
	if err := b.UpdateTrainer(context.Background(), trainer, g, 1e20); err != nil {
		t.Fatal(err)
	}
	wantGrads := trainingStepGradientMap(wg)
	for i, name := range trainer.names {
		assertSliceClose(t, "step padding parameter "+name, d.validatedGradients[i], wantGrads[name], 5e-5)
	}
}
func TestTrainingResidentDeviceScaleClippedUpdates(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := NewFullTrainer(cpuLM, cpuFlow, cpuW, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	backend, _ := installResidentGraph(t, trainer, "vulkan-resident-devicescale-hybrid-v1")
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	config := DefaultTrainingStepConfig()
	for step, ceiling := range []float32{.01, .1, 1e20} {
		_, cpuGrad, err := PocketTrainingStep(cpuLM, cpuFlow, cpuW, batch, samples, config)
		if err != nil {
			t.Fatal(err)
		}
		if err := backend.PrepareTrainingStep(context.Background(), trainer); err != nil {
			t.Fatal(err)
		}
		got, g, err := PocketTrainingStep(lm, flow, w, batch, samples, config)
		if err != nil {
			t.Fatal(err)
		}
		if !finite64(got.Loss) {
			t.Fatal("loss")
		}
		// Independent F64 direct norm over CPU gradients, then one F32 scale.
		gradients := trainingStepGradientMap(cpuGrad)
		var norm float64
		for _, name := range reference.names {
			for _, v := range gradients[name] {
				norm = math.Hypot(norm, float64(v))
			}
		}
		scale := float32(1)
		if norm > float64(ceiling) {
			scale = float32(float64(ceiling) / norm)
		}
		for _, x := range gradients {
			for i := range x {
				x[i] *= scale
			}
		}
		if err := reference.Step(cpuGrad); err != nil {
			t.Fatal(err)
		}
		if err := backend.UpdateTrainer(context.Background(), trainer, g, ceiling); err != nil {
			t.Fatal(err)
		}
		for name, p := range reference.params {
			assertSliceClose(t, "clipped device update "+name, trainer.params[name], p, 4e-6)
			assertSliceClose(t, "clipped device M "+name, trainer.Moments[name].M, reference.Moments[name].M, 5e-6)
			assertSliceClose(t, "clipped device V "+name, trainer.Moments[name].V, reference.Moments[name].V, 5e-6)
			assertSliceClose(t, "clipped device EMA "+name, trainer.EMA[name], reference.EMA[name], 4e-6)
		}
		if err := backend.ValidateBoundary(trainer); err != nil {
			t.Fatal(err)
		}
		t.Logf("step=%d independent norm=%g ceiling=%g scale=%g", step+1, norm, ceiling, scale)
	}
	if err := backend.Close(); err != nil {
		t.Fatal(err)
	}
}

// Mirror the new kernel schedule: no host clipping constant is accepted. The
// globalClip buffer is read at dispatch, after diagnostic admission.
func (d *hostResidentGraphDriver) ValidateScaledUpdate(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	if step.GradientScale != 1 || len(clip.(*hostOptimizerBuffer).values) != 4 {
		return errors.New("device-scale admission")
	}
	return d.ValidateUpdate(ctx, p, g, m, v, ema, n, step)
}
func (d *hostResidentGraphDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	if step.GradientScale != 1 {
		return errors.New("host scale injected")
	}
	step.GradientScale = clip.(*hostOptimizerBuffer).values[3]
	return d.hostOptimizerDriver.Update(ctx, p, g, m, v, ema, n, step)
}
