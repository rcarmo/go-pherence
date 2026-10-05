//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestTrainingResidentPrimaryRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-primaryfused-hybrid-v1", "vulkan-resident-primaryfused-hybrid-v1"} {
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
func TestTrainingResidentPrimaryFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
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
func TestTrainingResidentPrimaryObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-primaryfused-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentPrimaryAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
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
	r, b, _ = residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
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
func TestTrainingResidentPrimaryInputAdmission(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "budget"} {
		r, b, _ := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
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
func TestTrainingResidentPrimaryPaddingParity(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-primaryfused-hybrid-v1")
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
func TestTrainingResidentPrimaryClippedUpdates(t *testing.T) {
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
	backend, _ := installResidentGraph(t, trainer, "vulkan-resident-primaryfused-hybrid-v1")
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

// Contaminate host packing scratch after the first tape execution without
// replacing the resident owner. The new path must neither consume nor rebuild it.

func TestTrainingResidentPrimaryAccounting(t *testing.T) {
	legacy, _, ld := residentGraphRunner(t, "ptx-resident-endpointfused-hybrid-v1")
	fused, _, fd := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
	if _, err := legacy.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := fused.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for n, x := range legacy.trainer.params {
		assertSliceClose(t, n, fused.trainer.params[n], x, 4e-6)
		assertSliceClose(t, n+" M", fused.trainer.Moments[n].M, legacy.trainer.Moments[n].M, 5e-6)
		assertSliceClose(t, n+" V", fused.trainer.Moments[n].V, legacy.trainer.Moments[n].V, 5e-6)
		assertSliceClose(t, n+" EMA", fused.trainer.EMA[n], legacy.trainer.EMA[n], 4e-6)
	}
	if ld.tapes-fd.tapes != 2 || ld.downloads-fd.downloads != 10 || ld.accumulations != fd.accumulations {
		t.Fatal("primary accounting", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.accumulations, fd.accumulations)
	}
	t.Logf("endpointfused vs primaryfused tapes %d -> %d,downloads %d -> %d,commands %d -> %d,logical bytes %d -> %d,planned owned %d -> %d;native latency unmeasured", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.graphCommands, fd.graphCommands, ld.graphBytes, fd.graphBytes, ld.pooledOwnedBytes, fd.pooledOwnedBytes)
}
func TestTrainingResidentPrimaryScalePolicy(t *testing.T) {
	var raw PrimaryTrainingResult
	for _, scale := range []float32{0, .25, 1} {
		t.Run(fmt.Sprint(scale), func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
			m := r.trainer.Flow
			in := PrimaryTrainingInputs{Condition: make([]float32, m.Condition.In), Input: make([]float32, m.Input.In), GV: make([]float32, m.Input.In), GT: make([]float32, m.Input.In), EndpointInput: make([]float32, m.Input.In), Target: make([]float32, m.Input.In), Noise: make([]float32, m.Input.In), S: .25, T: .8, DDelta: .03}
			for i := range in.GV {
				in.GV[i] = float32(i+1) * .1
				in.GT[i] = float32(i+1) * .02
				in.EndpointInput[i] = .05
				in.Target[i] = .2
				in.Noise[i] = -.1
			}
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			p, err := m.BuildPrimaryTrainingTape(in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			want := interpretFlowTape(t, p.Tape)
			before, download := d.accumulations, d.downloads
			accumulated := [][]float32{}
			d.afterAccumulate = func(v []float32) { accumulated = append(accumulated, append([]float32(nil), v...)) }
			got, err := b.EvaluatePrimary(m, in, scale)
			if err != nil {
				t.Fatal(err)
			}
			for i, x := range [][]float32{got.DCondition, got.DTarget, got.DNoise, got.DTimes} {
				assertSliceClose(t, "raw primary VJP", x, want[i], 2e-5)
			}
			if scale == 0 {
				raw = got
			} else {
				for i, x := range [][]float32{got.DCondition, got.DTarget, got.DNoise, got.DTimes} {
					assertSliceClose(t, "scale-independent inputs", x, [][]float32{raw.DCondition, raw.DTarget, raw.DNoise, raw.DTimes}[i], 0)
				}
			}
			if d.accumulations-before != len(p.ResidentBindings) || d.downloads-download != 4 {
				t.Fatal("singleaccum/boundary")
			}
			if len(accumulated) != len(p.ResidentBindings) {
				t.Fatal("gradient capture")
			}
			for i, v := range accumulated {
				expected := append([]float32(nil), want[i+4]...)
				for j := range expected {
					expected[j] *= scale
				}
				assertSliceClose(t, "parameter scaled once", v, expected, 2e-5)
			}
			if _, err := b.state.Snapshot(context.Background()); err == nil {
				t.Fatal("incomplete gradient boundary snapshot")
			}

		})
	}
}
func TestTrainingResidentPrimaryDirectAdmission(t *testing.T) {
	for _, kind := range []string{"inactive", "scope", "owner", "shape", "scale", "dispatch", "post", "resultcount"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-primaryfused-hybrid-v1")
			m := r.trainer.Flow
			in := PrimaryTrainingInputs{Condition: make([]float32, m.Condition.In), Input: make([]float32, m.Input.In), GV: make([]float32, m.Input.In), GT: make([]float32, m.Input.In), EndpointInput: make([]float32, m.Input.In), Target: make([]float32, m.Input.In), Noise: make([]float32, m.Input.In), S: .25, T: .8}
			scale := float32(.25)
			if kind != "inactive" {
				if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
					t.Fatal(err)
				}
			}
			switch kind {
			case "scope":
				b.Scope = "ptx-resident-endpointfused-hybrid-v1"
			case "owner":
				m.BranchBackend = nil
			case "shape":
				in.GT = nil
			case "scale":
				scale = -1
			case "dispatch":
				d.failTape = 1
			case "post":
				d.afterDownload = func(v [][]float32) { v[0][0] = float32(math.NaN()) }
			case "resultcount":
				d.afterDownload = func(v [][]float32) { v[3] = nil }
			}
			if _, err := b.EvaluatePrimary(m, in, scale); err == nil {
				t.Fatal("invalid source/result")
			}
			if kind == "inactive" {
				return
			}
			if b.terminal == nil {
				t.Fatal("nonterminal")
			}
			tapes := d.tapes
			if _, err := b.EvaluatePrimary(m, in, scale); err == nil || d.tapes != tapes {
				t.Fatal("retry")
			}
			if b.ValidateBoundary(r.trainer) == nil || b.Close() == nil || b.state == nil {
				t.Fatal("uncertain owner freed")
			}
		})
	}
}
