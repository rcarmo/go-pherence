//go:build linux

package pockettts

import (
	"context"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestTrainingStagedBackboneReference(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	lm, flow, w := r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting
	batch := r.rows[0].Batch
	frames, h, n := batch.Frames, lm.Hidden, lm.LatentDim
	mask := make([]bool, frames)
	for i := range mask {
		mask[i] = i < r.rows[0].ValidFrames
	}
	original, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	reference := interpretFlowTape(t, original.Tape)
	ft, loss, err := b.evaluateStagedForward(lm, batch, mask, .1)
	if err != nil {
		t.Fatal(err)
	}
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	assertSliceClose(t, "staged forward hidden", ft.output.Z, reference[1][prefix*h:], 4e-6)
	assertSliceClose(t, "staged forward EOS", []float32{float32(loss)}, reference[len(reference)-1], 2e-6)
	hidden := ft.output.Z
	if err = b.closeFlowHandoff(); err != nil {
		t.Fatal(err)
	}
	in := StepReductionInputs{Frames: frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, frames*h), DistillCondition: make([]float32, frames*h), DiagonalTarget: make([]float32, frames*n), DistillTarget: make([]float32, frames*n), DiagonalLogvar: make([]float32, frames), DistillLogvar: make([]float32, frames), RowMetrics: make([]float32, frames*4), EOS: loss}
	for row, active := range mask {
		if active {
			for j := 0; j < h; j++ {
				in.DiagonalCondition[row*h+j] = .001 * float32(j+1)
				in.DistillCondition[row*h+j] = -.002 * float32(j+1)
			}
			for j := 0; j < n; j++ {
				in.DiagonalTarget[row*n+j] = .003
				in.DistillTarget[row*n+j] = -.001
			}
		}
	}
	samples := WeightSampleInputs{DiagonalTime: make([]float32, frames), DistillS: make([]float32, frames), DistillT: make([]float32, frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	for i := range mask {
		if mask[i] {
			samples.DiagonalTime[i] = .3
			samples.DistillS[i] = .3
			samples.DistillT[i] = .8
		}
	}
	cfg := DefaultTrainingStepConfig()
	full, err := lm.buildStepBackboneTrainingTape(w, batch, hidden, in, cfg, samples, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	all := interpretFlowTape(t, full.Tape)
	count := 6
	if batch.VoiceFrames > 0 {
		count = 7
	}
	want, err := decodeStepBackboneBoundary(all[:count], batch, hidden, in, samples)
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.evaluateStagedStepBackbone(lm, flow, w, batch, hidden, in, cfg, samples)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "staged normalized", got.Inputs.NormalizedLatents, want.Inputs.NormalizedLatents, 8e-6)
	assertSliceClose(t, "staged voice", got.Inputs.VoiceLatents, want.Inputs.VoiceLatents, 8e-6)
	assertSliceClose(t, "staged logvar", got.Reduction.DiagonalLogvar, want.Reduction.DiagonalLogvar, 6e-8)
	expected := map[string][]float32{}
	for name, binding := range full.ResidentBindings {
		for i, id := range full.Tape.Outputs {
			if id == binding.Gradient {
				expected[name] = all[i]
			}
		}
	}
	zeros := map[string][]float32{}
	for _, binding := range r.trainer.bindings {
		if _, ok := expected[binding.name]; !ok {
			zeros[binding.name] = make([]float32, len(binding.parameter))
		}
	}
	if err = b.state.Accumulate(b.ctx, zeros, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = b.state.Update(b.ctx, 100000); err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(r.trainer.params))
	for name := range r.trainer.params {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		if want, ok := expected[name]; ok {
			assertSliceClose(t, "staged accumulated "+name, d.validatedGradients[i], want, 1e-5)
		}
	}
	if b.transformerStages != nil || b.stagedBackbone != nil || b.handoffLease != nil {
		t.Fatal("retained successful stage boundary")
	}
	if err = b.state.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingStagedBackboneFailureRetention(t *testing.T) {
	base, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), base.trainer); err != nil {
		t.Fatal(err)
	}
	batch := base.rows[0].Batch
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < base.rows[0].ValidFrames
	}
	if _, _, err := b.evaluateStagedForward(base.trainer.FlowLM, batch, mask, .1); err != nil {
		t.Fatal(err)
	}
	total := d.tapes
	// Fault every submitted forward-stage construction; no checkpoint or retry.
	for at := 1; at <= total; at++ {
		t.Run(fmt.Sprint(at), func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
			before := mustFullState(t, r.trainer)
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			d.failTape = at
			if _, _, err := b.evaluateStagedForward(r.trainer.FlowLM, batch, mask, .1); err == nil || b.terminal == nil {
				t.Fatal("failed stage accepted", at)
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
				t.Fatal("failed forward mutated trainer")
			}
			calls := d.tapes
			if _, _, err := b.evaluateStagedForward(r.trainer.FlowLM, batch, mask, .1); err == nil || d.tapes != calls {
				t.Fatal("forward retry")
			}
			if b.Close() == nil {
				t.Fatal("failed owner closed")
			}
		})
	}
}

func stagedBackboneFailureFixture(t *testing.T) (*TrainingRunner, *ResidentTrainingBackend, *hostResidentGraphDriver, FlowLMTrainingBatch, []float32, StepReductionInputs, WeightSampleInputs) {
	t.Helper()
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	batch := r.rows[0].Batch
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < r.rows[0].ValidFrames
	}
	ft, loss, err := b.evaluateStagedForward(r.trainer.FlowLM, batch, mask, .1)
	if err != nil {
		t.Fatal(err)
	}
	if err = b.closeFlowHandoff(); err != nil {
		t.Fatal(err)
	}
	h, n := r.trainer.FlowLM.Hidden, r.trainer.FlowLM.LatentDim
	frames := batch.Frames
	in := StepReductionInputs{Frames: frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, frames*h), DistillCondition: make([]float32, frames*h), DiagonalTarget: make([]float32, frames*n), DistillTarget: make([]float32, frames*n), DiagonalLogvar: make([]float32, frames), DistillLogvar: make([]float32, frames), RowMetrics: make([]float32, frames*4), EOS: loss}
	samples := WeightSampleInputs{DiagonalTime: make([]float32, frames), DistillS: make([]float32, frames), DistillT: make([]float32, frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	return r, b, d, batch, ft.output.Z, in, samples
}

func TestTrainingStagedBackwardFailureRetention(t *testing.T) {
	r, b, d, batch, hidden, in, samples := stagedBackboneFailureFixture(t)
	start := d.tapes
	if _, err := b.evaluateStagedStepBackbone(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, batch, hidden, in, DefaultTrainingStepConfig(), samples); err != nil {
		t.Fatal(err)
	}
	calls := d.tapes - start
	for at := 1; at <= calls; at++ {
		t.Run(fmt.Sprint(at), func(t *testing.T) {
			r, b, d, batch, hidden, in, samples := stagedBackboneFailureFixture(t)
			before := mustFullState(t, r.trainer)
			d.failTape = d.tapes + at
			if _, err := b.evaluateStagedStepBackbone(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, batch, hidden, in, DefaultTrainingStepConfig(), samples); err == nil || b.terminal == nil {
				t.Fatal("failed reverse accepted", at)
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
				t.Fatal("failed reverse published mutation")
			}
			count := d.tapes
			if _, err := b.evaluateStagedStepBackbone(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, batch, hidden, in, DefaultTrainingStepConfig(), samples); err == nil || d.tapes != count {
				t.Fatal("reverse retry")
			}
			if b.ValidateBoundary(r.trainer) == nil || b.Close() == nil {
				t.Fatal("failed reverse owner published/closed")
			}
		})
	}
	for _, kind := range []string{"hidden", "batch", "mask", "weight", "parameter"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d, batch, hidden, in, samples := stagedBackboneFailureFixture(t)
			before := d.tapes
			cfg := DefaultTrainingStepConfig()
			switch kind {
			case "hidden":
				hidden[0] += 1
			case "batch":
				batch.NormalizedLatents = append([]float32(nil), batch.NormalizedLatents...)
				batch.NormalizedLatents[0] += 1
			case "mask":
				in.Mask = append([]bool(nil), in.Mask...)
				in.Mask[0] = !in.Mask[0]
			case "weight":
				cfg.EOSLossWeight = .2
			case "parameter":
				r.trainer.FlowLM.Transformer.Layers[0].FC1.Weight[0] += 1
			}
			if _, err := b.evaluateStagedStepBackbone(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, batch, hidden, in, cfg, samples); err == nil || b.terminal == nil {
				t.Fatal("changed stage provenance accepted", kind)
			}
			if d.tapes != before {
				t.Fatal("changed provenance reached native work", kind)
			}
		})
	}
}

func TestTrainingStagedLongFullStepCPUReference(t *testing.T) {
	testTrainingStagedLongFullStepCPUReference(t, 100000, false)
}
func TestTrainingStagedLongClippedStepCPUReference(t *testing.T) {
	testTrainingStagedLongFullStepCPUReference(t, 1, false)
}
func TestTrainingStagedLongCheckpointResume(t *testing.T) {
	testTrainingStagedLongFullStepCPUReference(t, 1, true)
}
func testTrainingStagedLongFullStepCPUReference(t *testing.T, maxNorm float32, replay bool) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	refLM, refFlow, refW := trainingStepModelsFromOracle(t, oracle)
	frames := 129
	channels := lm.LatentDim
	batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: oracle.VoiceFrames, TextTokens: append([]uint32(nil), oracle.TextTokens...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), NormalizedLatents: make([]float32, frames*channels)}
	samples := TrainingStepSamples{Mask: make([]bool, frames), Noise: make([]float32, frames*channels), DiagonalTime: make([]float32, frames), DistillS: make([]float32, frames), DistillT: make([]float32, frames)}
	valid := frames - 1
	for row := 0; row < valid; row++ {
		samples.Mask[row] = true
		samples.DiagonalTime[row] = .3
		samples.DistillS[row] = .3
		samples.DistillT[row] = .8
		for c := 0; c < channels; c++ {
			batch.NormalizedLatents[row*channels+c] = oracle.NormalizedLatents[(row%2)*channels+c]
			samples.Noise[row*channels+c] = oracle.Noise[(row%2)*channels+c]
		}
	}
	adam := AdamWConfig{LearningRate: oracle.AdamW.LearningRate, Beta1: oracle.AdamW.Beta1, Beta2: oracle.AdamW.Beta2, Epsilon: oracle.AdamW.Epsilon, WeightDecay: oracle.AdamW.WeightDecay}
	ref, err := NewFullTrainer(refLM, refFlow, refW, adam, oracle.AdamW.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	want, wantGrad, err := PocketTrainingStep(refLM, refFlow, refW, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err = (&TrainingRunner{trainer: ref, maxNorm: maxNorm}).clipGradients(wantGrad); err != nil {
		t.Fatal(err)
	}
	if err = ref.Step(wantGrad); err != nil {
		t.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, w, adam, oracle.AdamW.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := installResidentGraph(t, trainer, "vulkan-resident-devicehandoff-hybrid-v1")
	if err = b.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	// Preparation uses the real legacy host driver's capabilities. Selection
	// switches only after admission so this fixture cannot pretend shared support.
	b.Scope = "vulkan-resident-sharedworkspace-hybrid-v1"
	got, grad, err := PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "staged long metrics", []float32{float32(got.FlowDiagonal), float32(got.FlowDistill), float32(got.FlowLoss), float32(got.EOS), float32(got.Loss)}, []float32{float32(want.FlowDiagonal), float32(want.FlowDistill), float32(want.FlowLoss), float32(want.EOS), float32(want.Loss)}, 3e-5)
	assertSliceClose(t, "staged long normalized", grad.Inputs.NormalizedLatents, wantGrad.Inputs.NormalizedLatents, 5e-5)
	assertSliceClose(t, "staged long voice", grad.Inputs.VoiceLatents, wantGrad.Inputs.VoiceLatents, 5e-5)
	// Restore the actual owner scope before completed snapshot, which this host
	// mock supports. No shader/native qualification is inferred from this test.
	b.Scope = "vulkan-resident-devicehandoff-hybrid-v1"
	if err = b.UpdateTrainer(context.Background(), trainer, grad, maxNorm); err != nil {
		t.Fatal(err)
	}
	for name, values := range trainer.params {
		assertSliceClose(t, "staged long parameter "+name, values, ref.params[name], 4e-6)
		assertSliceClose(t, "staged long EMA "+name, trainer.EMA[name], ref.EMA[name], 4e-6)
	}
	if b.stagedBackbone != nil || b.transformerStages != nil || b.handoffLease != nil {
		t.Fatal("long step boundary retained")
	}
	if err = b.Close(); err != nil {
		t.Fatal(err)
	}
	if replay {
		state, err := trainer.State()
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "staged-step-1.model.bin")
		if err = SaveFullTrainingStateBinary(path, state); err != nil {
			t.Fatal(err)
		}
		loaded, err := LoadFullTrainingStateBinary(path, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		freshLM, freshFlow, freshW := trainingStepModelsFromOracle(t, oracle)
		fresh, err := NewFullTrainer(freshLM, freshFlow, freshW, adam, oracle.AdamW.EMADecay)
		if err != nil {
			t.Fatal(err)
		}
		if err = fresh.LoadState(loaded); err != nil {
			t.Fatal(err)
		}
		freshBackend, _ := installResidentGraph(t, fresh, "vulkan-resident-devicehandoff-hybrid-v1")
		if err = freshBackend.PrepareTrainingStep(context.Background(), fresh); err != nil {
			t.Fatal(err)
		}
		freshBackend.Scope = "vulkan-resident-sharedworkspace-hybrid-v1"
		_, actualGrad, err := PocketTrainingStep(freshLM, freshFlow, freshW, batch, samples, DefaultTrainingStepConfig())
		if err != nil {
			t.Fatal(err)
		}
		freshBackend.Scope = "vulkan-resident-devicehandoff-hybrid-v1"
		if err = freshBackend.UpdateTrainer(context.Background(), fresh, actualGrad, maxNorm); err != nil {
			t.Fatal(err)
		}
		_, referenceGrad, err := PocketTrainingStep(refLM, refFlow, refW, batch, samples, DefaultTrainingStepConfig())
		if err != nil {
			t.Fatal(err)
		}
		if err = (&TrainingRunner{trainer: ref, maxNorm: maxNorm}).clipGradients(referenceGrad); err != nil {
			t.Fatal(err)
		}
		if err = ref.Step(referenceGrad); err != nil {
			t.Fatal(err)
		}
		for name, values := range fresh.params {
			assertSliceClose(t, "staged resumed parameter "+name, values, ref.params[name], 4e-6)
			assertSliceClose(t, "staged resumed EMA "+name, fresh.EMA[name], ref.EMA[name], 4e-6)
			assertSliceClose(t, "staged resumed M "+name, fresh.Moments[name].M, ref.Moments[name].M, 4e-6)
			assertSliceClose(t, "staged resumed V "+name, fresh.Moments[name].V, ref.Moments[name].V, 4e-6)
		}
		if fresh.StepCount != 2 || freshBackend.ValidateBoundary(fresh) != nil {
			t.Fatal("resumed staged boundary")
		}
		if err = freshBackend.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
