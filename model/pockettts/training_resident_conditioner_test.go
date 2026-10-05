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

func TestTrainingResidentConditionerRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-conditioner-hybrid-v1", "vulkan-resident-conditioner-hybrid-v1"} {
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
func TestTrainingResidentConditionerFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
func TestTrainingResidentConditionerObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-conditioner-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentConditionerAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
	r, b, _ = residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
func TestTrainingResidentConditionerInputAdmission(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "budget"} {
		r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
func TestTrainingResidentConditionerPaddingParity(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-conditioner-hybrid-v1")
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
func TestTrainingResidentConditionerClippedUpdates(t *testing.T) {
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
	backend, _ := installResidentGraph(t, trainer, "vulkan-resident-conditioner-hybrid-v1")
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
func TestTrainingResidentConditionerPackingBypass(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	oracle := loadTrainingStepOracle(t)
	batch := FlowLMTrainingBatch{TextTokens: oracle.TextTokens, VoiceFrames: oracle.VoiceFrames, VoiceLatents: oracle.VoiceLatents, NormalizedLatents: oracle.NormalizedLatents, Frames: oracle.Frames}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	workspace, err := NewTrainingWorkspace(r.trainer.FlowLM, r.trainer.Flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	if err = b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	injected := false
	d.afterRun = func() {
		if !injected {
			for _, x := range [][]float32{workspace.weightInputs, workspace.dLogvars} {
				for i := range x {
					x[i] = float32(math.NaN())
				}
			}
			injected = true
		}
	}
	if _, _, err = PocketTrainingStepInto(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, batch, samples, DefaultTrainingStepConfig(), workspace); err != nil {
		t.Fatal(err)
	}
	if !injected {
		t.Fatal("no tape execution")
	}
	for _, x := range [][]float32{workspace.weightInputs, workspace.dLogvars} {
		for _, v := range x {
			if !math.IsNaN(float64(v)) {
				t.Fatal("host packing scratch consumed/rebuilt")
			}
		}
	}
	d.afterRun = nil
}
func TestTrainingResidentConditionerFailureAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	if _, err := b.EvaluateSampleWeight(r.trainer.Weighting, WeightSampleInputs{}, false); err == nil {
		t.Fatal("inactive")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	before := mustFullState(t, r.trainer)
	s := WeightSampleInputs{[]float32{.2}, []float32{.1}, []float32{.8}, []float32{1}, nil}
	if _, err := b.EvaluateSampleWeight(r.trainer.Weighting, s, true); err == nil || b.terminal == nil {
		t.Fatal("invalid column")
	}
	if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || b.Close() == nil {
		t.Fatal("admission changed host state/owner")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-devicescale-hybrid-v1")
	if lb.SampleWeightEnabled() {
		t.Fatal("legacy opted in")
	}
	if err := lb.PrepareTrainingStep(context.Background(), legacy.trainer); err != nil {
		t.Fatal(err)
	}
	s.DistillSeed = []float32{1}
	if _, err := lb.EvaluateSampleWeight(legacy.trainer.Weighting, s, true); err == nil || lb.terminal == nil {
		t.Fatal("legacy scope injection")
	}
}
func TestTrainingResidentConditionerAccounting(t *testing.T) {
	legacy, _, ld := residentGraphRunner(t, "ptx-resident-voicefused-hybrid-v1")
	fused, _, fd := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
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
	if ld.tapes-fd.tapes != 2 || ld.downloads-fd.downloads != 2 || ld.accumulations != fd.accumulations {
		t.Fatal("EOS boundary/accumulation accounting", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.accumulations, fd.accumulations)
	}
	t.Logf("conditioner vs voicefused tapes %d -> %d,downloads %d -> %d,commands %d -> %d,logical bytes %d -> %d,planned owned %d -> %d;native latency unmeasured", ld.tapes, fd.tapes, ld.downloads, fd.downloads, ld.graphCommands, fd.graphCommands, ld.graphBytes, fd.graphBytes, ld.pooledOwnedBytes, fd.pooledOwnedBytes)
}
func TestTrainingResidentConditionerSeededNoAccumulation(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	s := WeightSampleInputs{[]float32{.3, .7}, []float32{.1, .2}, []float32{.6, .8}, []float32{.2, -.1}, []float32{-.3, .4}}
	plan, err := r.trainer.Weighting.BuildSampleWeightTrainingTape(s, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	want, err := plan.Decode(interpretFlowTape(t, plan.Tape))
	if err != nil {
		t.Fatal(err)
	}
	before := d.accumulations
	got, err := b.EvaluateSampleWeight(r.trainer.Weighting, s, false)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "seeded nonaccum output", got.Output, want.Output, 2e-7)
	assertSliceClose(t, "seeded nonaccum inputVJP", got.DInput, want.DInput, 2e-6)
	if d.accumulations != before {
		t.Fatal("nonaccum parameter adjoints")
	}
	for _, x := range [][]float32{s.DiagonalTime, s.DistillS, s.DistillT, s.DiagonalSeed, s.DistillSeed} {
		if f32SlicesOverlap(x, got.Output) || f32SlicesOverlap(x, got.DInput) {
			t.Fatal("borrowed raw column")
		}
	}
	for _, x := range [][]float32{s.DiagonalSeed, s.DistillSeed} {
		clear(x)
	}
	forward, err := b.EvaluateSampleWeight(r.trainer.Weighting, s, false)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "forward", forward.Output, want.Output, 2e-7)
	if !zeroTrainingSeed(forward.DInput) || d.accumulations != before {
		t.Fatal("forward reverse")
	}
	// Accumulation with zero seeds must retain every parameter gradient binding;
	// all parameters must be marked touched rather than silently missing later.
	if _, err = b.EvaluateSampleWeight(r.trainer.Weighting, s, true); err != nil {
		t.Fatal(err)
	}
	if d.accumulations == before {
		t.Fatal("zero seed accumulated gradient removed")
	}
}

func TestTrainingResidentConditionerBoundaryGuards(t *testing.T) {
	for _, category := range []string{"shape", "seed", "scope", "owner"} {
		t.Run(category, func(t *testing.T) {
			r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
			o := loadTrainingStepOracle(t)
			batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, TextTokens: o.TextTokens, VoiceLatents: o.VoiceLatents, NormalizedLatents: o.NormalizedLatents}
			zeroZ, zeroEOS := make([]float32, o.Frames*r.trainer.FlowLM.Hidden), make([]float32, o.Frames)
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			tape, err := r.trainer.FlowLM.forwardTrainingTapeEOSMode(batch, zeroZ, zeroEOS, false, true)
			if err != nil {
				t.Fatal(err)
			}
			result, err := b.EvaluateEOSProjection(r.trainer.FlowLM, tape.output.Z, o.Mask, .1)
			if err != nil {
				t.Fatal(err)
			}
			tape.fusedEOSInput = result.DInput
			switch category {
			case "shape":
				tape.fusedEOSInput = nil
			case "seed":
				zeroEOS[0] = 1
			case "scope":
				b.Scope = "ptx-resident-zeroconst-hybrid-v1"
			case "owner":
				r.trainer.FlowLM.EOSBackend = nil
			}
			if _, _, err = r.trainer.FlowLM.backwardTrainingTape(tape, batch, zeroZ, zeroEOS); err == nil {
				t.Fatal("bad fused VJP seed/owner accepted")
			}
		})
	}
}

func TestTrainingResidentConditionerSeededBoundary(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	o := loadTrainingStepOracle(t)
	batch := FlowLMTrainingBatch{Frames: o.Frames, NormalizedLatents: o.NormalizedLatents}
	seed := make([]float32, batch.Frames*r.trainer.FlowLM.Hidden)
	for i := range seed {
		seed[i] = float32(i%7-3) * .1
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	p, err := r.trainer.FlowLM.BuildAudioProjectionTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	want := interpretFlowTape(t, p.Tape)
	before := d.accumulations
	got, err := b.EvaluateAudioProjection(r.trainer.FlowLM, batch, seed, false)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "nonaccum output", got.Output, want[0], 2e-7)
	assertSliceClose(t, "nonaccum normalized VJP", got.DNormalized, want[1], 2e-7)
	if d.accumulations != before {
		t.Fatal("nonaccum param leak")
	}
	clear(seed)
	got, err = b.EvaluateAudioProjection(r.trainer.FlowLM, batch, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	if !zeroTrainingSeed(got.DNormalized) || d.accumulations-before != 2 {
		t.Fatal("zero seed BOS/weight roots pruned")
	}
}
func TestTrainingResidentConditionerOwnerGuard(t *testing.T) {
	for _, kind := range []string{"scope", "owner"} {
		t.Run(kind, func(t *testing.T) {
			r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
			o := loadTrainingStepOracle(t)
			batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
			z, eos := make([]float32, o.Frames*r.trainer.FlowLM.Hidden), make([]float32, o.Frames)
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			tape, err := r.trainer.FlowLM.forwardTrainingTape(batch, z, eos)
			if err != nil {
				t.Fatal(err)
			}
			if tape.audioInput != nil || tape.conditioner == nil {
				t.Fatal("host shifted input materialized")
			}
			if kind == "scope" {
				b.Scope = "ptx-resident-eosfused-hybrid-v1"
			} else {
				r.trainer.FlowLM.SequenceBackend = nil
			}
			if _, _, err = r.trainer.FlowLM.backwardTrainingTape(tape, batch, z, eos); err == nil {
				t.Fatal("fused owner removal")
			}
		})
	}
}

func TestTrainingResidentConditionerSeededSequence(t *testing.T) {
	for _, voiceRows := range []int{0, 1} {
		t.Run(fmt.Sprintf("voice%d", voiceRows), func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
			m := r.trainer.FlowLM
			batch := FlowLMTrainingBatch{Frames: 1, VoiceFrames: voiceRows, VoiceLatents: make([]float32, voiceRows*m.LatentDim), TextTokens: []uint32{1, 1}}
			for i := range batch.VoiceLatents {
				batch.VoiceLatents[i] = float32(i+1) * .1
			}
			audio := make([]float32, m.Hidden)
			seed := make([]float32, (1+voiceRows+2+1)*m.Hidden)
			for i := range seed {
				seed[i] = float32(i%7-3) * .1
			}
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			p, err := m.BuildVoiceSequenceTrainingTape(batch, audio, seed, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			want := interpretFlowTape(t, p.Tape)
			before := d.accumulations
			got, err := b.EvaluateVoiceSequence(m, batch, audio, seed, false)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceClose(t, "sequence nonaccum", got.Sequence, want[0], 2e-7)
			index := 1
			if voiceRows > 0 {
				assertSliceClose(t, "raw voice nonaccum", got.DVoice, want[1], 2e-7)
				index++
			}
			assertSliceClose(t, "audio nonaccum", got.DAudio, want[index], 2e-7)
			if d.accumulations != before {
				t.Fatal("nonaccum parameter leak")
			}
			clear(seed)
			got, err = b.EvaluateVoiceSequence(m, batch, audio, seed, true)
			if err != nil {
				t.Fatal(err)
			}
			if !zeroTrainingSeed(got.DVoice) || !zeroTrainingSeed(got.DAudio) || d.accumulations-before != 3 {
				t.Fatal("zero-seed speaker/embedding/BOS contribution missing")
			}
			// Even without voice rows the speaker binding stays present as a zero VJP.
			for _, binding := range p.ResidentBindings {
				if binding.Parameter < 0 || binding.Gradient < 0 {
					t.Fatal("missing source binding")
				}
			}
		})
	}
}
func TestTrainingResidentConditionerSequenceOwnerGuard(t *testing.T) {
	for _, kind := range []string{"scope", "owner"} {
		t.Run(kind, func(t *testing.T) {
			r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
			o := loadTrainingStepOracle(t)
			batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
			z, eos := make([]float32, o.Frames*r.trainer.FlowLM.Hidden), make([]float32, o.Frames)
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			tape, err := r.trainer.FlowLM.forwardTrainingTape(batch, z, eos)
			if err != nil {
				t.Fatal(err)
			}
			if tape.conditioner == nil {
				t.Fatal("fusion not captured")
			}
			if kind == "scope" {
				b.Scope = "ptx-resident-audiofused-hybrid-v1"
			} else {
				r.trainer.FlowLM.SequenceBackend = nil
			}
			if _, _, err = r.trainer.FlowLM.backwardTrainingTape(tape, batch, z, eos); err == nil {
				t.Fatal("voice fusion owner removal")
			}
		})
	}
}

func TestTrainingResidentConditionerSeededCombined(t *testing.T) {
	for _, v := range []int{0, 1} {
		t.Run(fmt.Sprintf("voice%d", v), func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
			m := r.trainer.FlowLM
			batch := FlowLMTrainingBatch{Frames: 1, VoiceFrames: v, VoiceLatents: make([]float32, v*m.LatentDim), NormalizedLatents: make([]float32, m.LatentDim), TextTokens: []uint32{1, 1}}
			for i := range batch.VoiceLatents {
				batch.VoiceLatents[i] = float32(i+1) * .1
			}
			seed := make([]float32, (1+v+2+1)*m.Hidden)
			for i := range seed {
				seed[i] = float32(i%7-3) * .1
			}
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			p, err := m.BuildConditionerTrainingTape(batch, seed, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			want := interpretFlowTape(t, p.Tape)
			before := d.accumulations
			got, err := b.EvaluateConditioner(m, batch, seed, false)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceClose(t, "combined sequence", got.Sequence, want[0], 2e-7)
			index := 1
			if v > 0 {
				assertSliceClose(t, "combined rawvoice", got.DVoice, want[1], 2e-7)
				index++
			}
			assertSliceClose(t, "combined normalized", got.DAudio, want[index], 2e-7)
			if d.accumulations != before {
				t.Fatal("seeded nonaccum parameter leak")
			}
			clear(seed)
			got, err = b.EvaluateConditioner(m, batch, seed, true)
			if err != nil {
				t.Fatal(err)
			}
			if !zeroTrainingSeed(got.DVoice) || !zeroTrainingSeed(got.DAudio) || d.accumulations-before != 5 {
				t.Fatal("zero-seed five param bindings pruned")
			}
			for _, values := range [][]float32{batch.VoiceLatents, batch.NormalizedLatents, seed} {
				if f32SlicesOverlap(got.DVoice, values) || f32SlicesOverlap(got.DAudio, values) {
					t.Fatal("borrowed boundary")
				}
			}
		})
	}
}
func TestTrainingResidentConditionerDirectAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-conditioner-hybrid-v1")
	if _, err := b.EvaluateConditioner(r.trainer.FlowLM, FlowLMTrainingBatch{}, nil, false); err == nil {
		t.Fatal("inactive")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateConditioner(r.trainer.FlowLM, FlowLMTrainingBatch{Frames: 1}, nil, true); err == nil || b.terminal == nil {
		t.Fatal("bad shape not terminal")
	}
	if b.Close() == nil {
		t.Fatal("failed active owner freed")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-voicefused-hybrid-v1")
	if err := lb.PrepareTrainingStep(context.Background(), legacy.trainer); err != nil {
		t.Fatal(err)
	}
	if lb.ConditionerEnabled() {
		t.Fatal("legacy enabled")
	}
	if _, err := lb.EvaluateConditioner(legacy.trainer.FlowLM, FlowLMTrainingBatch{}, nil, false); err == nil || lb.terminal == nil {
		t.Fatal("legacy injection")
	}
}
