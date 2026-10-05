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

func TestTrainingResidentDistillResultsRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-distillresults-hybrid-v1", "vulkan-resident-distillresults-hybrid-v1"} {
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
func TestTrainingResidentDistillResultsFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
func TestTrainingResidentDistillResultsObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-distillresults-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentDistillResultsAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
	r, b, _ = residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
func TestTrainingResidentDistillResultsInputAdmission(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "budget"} {
		r, b, _ := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
func TestTrainingResidentDistillResultsPaddingParity(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-distillresults-hybrid-v1")
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
func TestTrainingResidentDistillResultsClippedUpdates(t *testing.T) {
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
	backend, _ := installResidentGraph(t, trainer, "vulkan-resident-distillresults-hybrid-v1")
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

func TestTrainingResidentDistillResultsAccounting(t *testing.T) {
	legacy, _, ld := residentGraphRunner(t, "ptx-resident-stepresults-hybrid-v1")
	fused, _, fd := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
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
	if ld.tapes != fd.tapes || ld.downloads-fd.downloads != 2 || ld.accumulations != fd.accumulations || ld.downloadedElements-fd.downloadedElements != 6 {
		t.Fatal("accounting", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.downloadedElements, fd.downloadedElements)
	}
	t.Logf("stepresults vs distillresults tapes %d -> %d,downloads %d -> %d,elements %d -> %d,commands %d -> %d,logical bytes %d -> %d,planned owned %d -> %d;native latency unmeasured", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.downloadedElements, fd.downloadedElements, ld.graphCommands, fd.graphCommands, ld.graphBytes, fd.graphBytes, ld.pooledOwnedBytes, fd.pooledOwnedBytes)
}
func TestTrainingResidentDistillResultsDirect(t *testing.T) {
	for _, scope := range []string{"ptx-resident-distillresults-hybrid-v1", "vulkan-resident-distillresults-hybrid-v1"} {
		for _, scale := range []float32{0, .25, 1} {
			t.Run(fmt.Sprintf("%s/%g", scope, scale), func(t *testing.T) {
				public, pb, pd := residentGraphRunner(t, scope)
				private, b, d := residentGraphRunner(t, scope)
				m := private.trainer.Flow
				in := residentDistillFixture(t, m, .3, .8)
				if err := b.PrepareTrainingStep(context.Background(), private.trainer); err != nil {
					t.Fatal(err)
				}
				if err := pb.PrepareTrainingStep(context.Background(), public.trainer); err != nil {
					t.Fatal(err)
				}
				before := d.accumulations
				got, err := b.evaluateDistillResults(m, in, scale)
				if err != nil {
					t.Fatal(err)
				}
				want, err := pb.EvaluateDistill(public.trainer.Flow, in, scale)
				if err != nil {
					t.Fatal(err)
				}
				assertSliceClose(t, "private condition", got.DCondition, want.DCondition, 2e-5)
				assertSliceClose(t, "private target", got.DTarget, want.DTarget, 2e-5)
				assertClose64(t, "private loss", got.Loss, want.Loss, 2e-6)
				assertClose64(t, "private square", got.RawSquare, want.RawSquare, 2e-6)
				if got.DLogVariance != want.DLogVariance || len(want.DNoise) != len(in.Noise) || len(want.DTimes) != 2 {
					t.Fatal("public contract")
				}
				if d.downloads != 4 || pd.downloads != 5 || d.accumulations-before != len(flowHeadGradientMap(newFlowHeadGradients(m))) {
					t.Fatal("once-only parameter roots", d.downloads, pd.downloads, d.accumulations-before)
				}
				// Mark non-flow roots ready with zero branches before update.
				// Include flow roots too at zero, preserving their accumulated values.
				zeros := map[string][]float32{}
				for name, x := range private.trainer.params {
					zeros[name] = make([]float32, len(x))
				}
				if err := b.state.Accumulate(context.Background(), zeros, 0); err != nil {
					t.Fatal(err)
				}
				if err := pb.state.Accumulate(context.Background(), zeros, 0); err != nil {
					t.Fatal(err)
				}
				// The coordinator exposes accumulated roots in the update validation hook.
				if _, err := b.state.Update(context.Background(), 1e20); err != nil {
					t.Fatal(err)
				}
				if _, err := pb.state.Update(context.Background(), 1e20); err != nil {
					t.Fatal(err)
				}
				for i, g := range d.validatedGradients {
					assertSliceClose(t, "primary-only scaled roots", g, pd.validatedGradients[i], 2e-5)
				}
			})
		}
	}
}
func TestTrainingResidentDistillResultsLSDRouting(t *testing.T) {
	for _, stepOnly := range []bool{false, true} {
		r, b, d := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
		in := residentDistillFixture(t, r.trainer.Flow, .3, .8)
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		var got LSDDistillGradients
		var err error
		if stepOnly {
			_, got, err = r.trainer.Flow.lsdDistillTrainingStepBackendScaled(b, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, .25)
		} else {
			_, got, err = r.trainer.Flow.LSDDistillTrainingBackend(b, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance)
		}
		if err != nil {
			t.Fatal(err)
		}
		if stepOnly {
			if got.Noise != nil || got.DS != 0 || got.DT != 0 || d.downloads != 10 {
				t.Fatal("private LSD policy", d.downloads)
			}
		} else if len(got.Noise) != len(in.Noise) || d.downloads != 11 {
			t.Fatal("public LSD policy", d.downloads)
		}
	}
}
func TestTrainingResidentDistillResultsOutputFailures(t *testing.T) {
	for _, kind := range []string{"inactive", "owner", "scope", "scale", "input", "budget", "witness-budget", "shape", "nan", "alias", "source", "stats", "witness", "witness-nan"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-distillresults-hybrid-v1")
			m := r.trainer.Flow
			in := residentDistillFixture(t, m, .3, .8)
			scale := float32(1)
			if kind == "inactive" {
				if _, err := b.evaluateDistillResults(m, in, scale); err == nil {
					t.Fatal("inactive")
				}
				return
			}
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "owner":
				m = &FlowHeadCPU{}
			case "scope":
				b.Scope = "ptx-resident-stepresults-hybrid-v1"
			case "scale":
				scale = -1
			case "input":
				in.Target = nil
			case "budget":
				b.MaxTapeBytes = 1
			case "witness-budget":
				p, err := m.buildDistillTrainingTape(in, 64<<20, true)
				if err != nil {
					t.Fatal(err)
				}
				var bytes int64
				for _, s := range p.Tape.Tensors {
					n, _ := s.Elements()
					bytes += int64(n) * 4
				}
				b.MaxTapeBytes = bytes
			case "witness-nan":
				d.afterDownload = func(v [][]float32) { v[3][0] = float32(math.NaN()) }
			case "shape":
				d.afterDownload = func(v [][]float32) { v[2] = nil }
			case "nan":
				d.afterDownload = func(v [][]float32) { v[0][0] = float32(math.NaN()) }
			case "alias":
				d.afterDownload = func(v [][]float32) { v[1] = v[0][:2] }
			case "source":
				d.afterDownload = func(v [][]float32) { v[1] = in.Target }
			case "stats":
				d.afterDownload = func(v [][]float32) { v[2][0] = -1 }
			case "witness":
				d.afterDownload = func(v [][]float32) { v[3][0] = 1 }
			}
			if _, err := b.evaluateDistillResults(m, in, scale); err == nil || b.terminal == nil {
				t.Fatal("bad result admitted")
			}
			tapes := d.tapes
			if _, err := b.evaluateDistillResults(m, in, scale); err == nil || d.tapes != tapes {
				t.Fatal("retry")
			}
			if b.ValidateBoundary(r.trainer) == nil || b.Close() == nil {
				t.Fatal("owner released")
			}
		})
	}
}

func residentDistillFixture(t *testing.T, m *FlowHeadCPU, s, tt float32) DistillTrainingInputs {
	t.Helper()
	c, x := make([]float32, m.Condition.In), []float32{.3, -.2}
	for i := range c {
		c[i] = float32(i+1) * .1
	}
	zero := make([]float32, len(x))
	p, err := m.BuildFlowDeviceTimeTrainingTape(c, []float32{s, tt}, x, 1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	advance, err := BuildBranchTrainingTape("advance", [][]float32{x, v[0], v[1]}, tt-s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	a := interpretFlowTape(t, advance)
	e, err := m.BuildFlowDeviceTimeTrainingTape(c, []float32{tt, tt}, a[0], -1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	ev := interpretFlowTape(t, e.Tape)
	return DistillTrainingInputs{Condition: c, PrimaryInput: x, EndpointInput: a[0], Prediction: ev[0], Desired: a[1], Velocity: v[0], Tangent: v[1], Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: s, T: tt, LogVariance: .2}
}
