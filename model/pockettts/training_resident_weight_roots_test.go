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

func TestTrainingResidentWeightRootsRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-weightroots-hybrid-v1", "vulkan-resident-weightroots-hybrid-v1"} {
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
func TestTrainingResidentWeightRootsFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
func TestTrainingResidentWeightRootsObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-weightroots-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentWeightRootsAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
	r, b, _ = residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
func TestTrainingResidentWeightRootsInputAdmission(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "budget"} {
		r, b, _ := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
func TestTrainingResidentWeightRootsPaddingParity(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-weightroots-hybrid-v1")
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
func TestTrainingResidentWeightRootsClippedUpdates(t *testing.T) {
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
	backend, _ := installResidentGraph(t, trainer, "vulkan-resident-weightroots-hybrid-v1")
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

func TestTrainingResidentWeightRootsAccounting(t *testing.T) {
	legacy, _, ld := residentGraphRunner(t, "ptx-resident-diagforward-hybrid-v1")
	fused, _, fd := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
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
	if ld.tapes != fd.tapes || ld.downloads-fd.downloads != 2 || ld.accumulations != fd.accumulations {
		t.Fatal("weight roots accounting", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.accumulations, fd.accumulations)
	}
	t.Logf("diagforward vs weightroots tapes %d -> %d,downloads %d -> %d,commands %d -> %d,logical bytes %d -> %d,planned owned %d -> %d;native latency unmeasured", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.graphCommands, fd.graphCommands, ld.graphBytes, fd.graphBytes, ld.pooledOwnedBytes, fd.pooledOwnedBytes)
}
func TestTrainingResidentWeightRootsReference(t *testing.T) {
	for _, scope := range []string{"ptx-resident-weightroots-hybrid-v1", "vulkan-resident-weightroots-hybrid-v1"} {
		for _, zero := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s-%t", scope, zero), func(t *testing.T) {
				r, b, d := residentGraphRunner(t, scope)
				m := r.trainer.Weighting
				s := WeightSampleInputs{[]float32{.3, .7}, []float32{.1, .2}, []float32{.6, .8}, []float32{.2, -.1}, []float32{-.3, .4}}
				if zero {
					clear(s.DiagonalSeed)
					clear(s.DistillSeed)
				}
				if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
					t.Fatal(err)
				}
				plan, err := m.BuildSampleWeightTrainingTape(s, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				want := interpretFlowTape(t, plan.Tape)
				_, _, f64 := sampleWeightOracle(m, s)
				accumulated := [][]float32{}
				d.afterAccumulate = func(v []float32) { accumulated = append(accumulated, append([]float32(nil), v...)) }
				before, downloads := d.accumulations, d.downloads
				if err := b.accumulateSampleWeight(m, s); err != nil {
					t.Fatal(err)
				}
				if d.accumulations-before != len(plan.ResidentBindings) || d.downloads != downloads || len(accumulated) != len(plan.ParameterNames) {
					t.Fatal("allroots/no downloads")
				}
				for i, name := range plan.ParameterNames {
					assertSliceClose(t, name, accumulated[i], want[i+2], 2e-6)
					for j, x := range accumulated[i] {
						expected := f64[name][j]
						if math.Abs(float64(x)-expected) > 2e-6+2e-6*math.Abs(expected) {
							t.Fatal("F64 gradient", name, j)
						}
					}
				}
				accumulated = nil
				if err := b.accumulateSampleWeight(m, s); err != nil {
					t.Fatal(err)
				}
				for i, v := range accumulated {
					expected := append([]float32(nil), want[i+2]...)
					for j := range expected {
						expected[j] *= 2
					}
					assertSliceClose(t, "repeat once", v, expected, 2e-6)
				}
				// Public seeded API still returns the real forward output and input VJP.
				got, err := b.EvaluateSampleWeight(m, s, false)
				if err != nil {
					t.Fatal(err)
				}
				assertSliceClose(t, "public output", got.Output, want[0], 2e-6)
				assertSliceClose(t, "public input VJP", got.DInput, want[1], 2e-6)
				for _, x := range [][]float32{s.DiagonalTime, s.DistillS, s.DistillT, s.DiagonalSeed, s.DistillSeed} {
					if f32SlicesOverlap(x, got.Output) || f32SlicesOverlap(x, got.DInput) {
						t.Fatal("public alias")
					}
				}
			})
		}
	}
}
func TestTrainingResidentWeightRootsDirectAdmission(t *testing.T) {
	for _, kind := range []string{"inactive", "scope", "owner", "frames", "seed", "budget", "dispatch", "close"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-weightroots-hybrid-v1")
			m := r.trainer.Weighting
			s := WeightSampleInputs{[]float32{.3}, []float32{.1}, []float32{.6}, []float32{.2}, []float32{-.3}}
			if kind != "inactive" {
				if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
					t.Fatal(err)
				}
			}
			switch kind {
			case "scope":
				b.Scope = "ptx-resident-diagforward-hybrid-v1"
			case "owner":
				m.TrainingBackend = nil
			case "frames":
				s.DiagonalTime = nil
			case "seed":
				s.DistillSeed[0] = float32(math.NaN())
			case "budget":
				b.MaxTapeBytes = 1
			case "dispatch":
				d.failTape = 1
			case "close":
				d.afterTapeClose = func() error { return fmt.Errorf("uncertain roots tape close") }
			}
			if err := b.accumulateSampleWeight(m, s); err == nil {
				t.Fatal("invalid source/close")
			}
			if kind == "inactive" {
				return
			}
			if b.terminal == nil {
				t.Fatal("nonterminal")
			}
			tapes := d.tapes
			if err := b.accumulateSampleWeight(m, s); err == nil || d.tapes != tapes {
				t.Fatal("retry")
			}
			if b.ValidateBoundary(r.trainer) == nil || b.Close() == nil || b.state == nil {
				t.Fatal("uncertain owner freed")
			}
		})
	}
}
