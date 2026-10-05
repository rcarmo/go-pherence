//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func tinyRunner(t *testing.T, experimental bool) (*TrainingRunner, error) {
	t.Helper()
	lm, flow, weighting := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	return NewTrainingRunner(trainer, runIdentityFixture(), []PreparedTrainingRow{{Batch: tinyFlowLMBatch(), ValidFrames: 2}}, [6]uint64{1, 2, 3, 4, 5, 6}, TrainingRandomLimits{TrainingSampleLimits{3, 6}, 20}, 1, experimental)
}
func requireRunner(t *testing.T) *TrainingRunner {
	t.Helper()
	r, err := tinyRunner(t, true)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func TestTrainingRunnerUpdateAndCancellation(t *testing.T) {
	r := requireRunner(t)
	before := append([]float32(nil), r.trainer.FlowLM.Input.Weight...)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.Step(ctx); err == nil {
		t.Fatal("cancelled update admitted")
	}
	step, _, _ := r.Progress()
	if step != 0 {
		t.Fatal("cancellation advanced state")
	}
	metrics, err := r.Step(context.Background())
	workspace := r.workspace
	if err != nil || !finite64(metrics.Loss) {
		t.Fatal(metrics, err)
	}
	step, _, _ = r.Progress()
	if step != 1 || r.trainer.StepCount != 1 {
		t.Fatal("update boundary mismatch")
	}
	changed := false
	for i, v := range before {
		if v != r.trainer.FlowLM.Input.Weight[i] {
			changed = true
		}
	}
	if !changed {
		t.Fatal("optimizer did not update parameters")
	}
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r.workspace != workspace {
		t.Fatal("fixed shape rebuilt workspace")
	}
}
func TestTrainingRunnerCheckpointResume(t *testing.T) {
	a := requireRunner(t)
	if _, err := a.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := a.Checkpoint(dir, "step1", limits); err != nil {
		t.Fatal(err)
	}
	b := requireRunner(t)
	if err := b.Restore(dir, limits); err != nil {
		t.Fatal(err)
	}
	for _, r := range []*TrainingRunner{a, b} {
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	pa, pb := a.trainer.params, b.trainer.params
	var square float64
	var count int
	for name, values := range pa {
		for i, v := range values {
			d := float64(v) - float64(pb[name][i])
			square += d * d
			count++
			if math.Abs(d) > 1e-6 {
				t.Fatalf("resume drift %s[%d]=%g", name, i, d)
			}
		}
	}
	if math.Sqrt(square/float64(count)) > 1e-7 {
		t.Fatal("resume RMS drift")
	}
	if !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, b.run)) {
		t.Fatal("raw state resume changed")
	}
	if err := b.Restore(dir, limits); err == nil {
		t.Fatal("restored into live runner")
	}
	if _, err := b.Checkpoint(dir, "step1", limits); err == nil {
		t.Fatal("generation overwritten")
	}
	if _, err := os.Stat(filepath.Join(dir, "step1.model.bin")); err != nil {
		t.Fatal("old generation removed")
	}
}
func TestTrainingRunnerRejectsExperimentalAndCorruption(t *testing.T) {
	if _, err := tinyRunner(t, false); err == nil {
		t.Fatal("experimental optin missing")
	}
	r := requireRunner(t)
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := r.Checkpoint(dir, "initial", limits); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(dir, "initial.model.bin"), os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.WriteAt([]byte{123}, 60); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	fresh := requireRunner(t)
	if err := fresh.Restore(dir, limits); err == nil {
		t.Fatal("corrupt checkpoint restored")
	}
	step, _, _ := fresh.Progress()
	if step != 0 || fresh.trainer.StepCount != 0 {
		t.Fatal("rejection mutated fresh runner")
	}
}
func TestTrainingRunnerMalformedTrainerAdmission(t *testing.T) {
	for _, trainer := range []*FullTrainer{nil, {}, {FlowLM: tinyFlowLMTraining()}} {
		if _, err := NewTrainingRunner(trainer, runIdentityFixture(), []PreparedTrainingRow{{Batch: tinyFlowLMBatch(), ValidFrames: 2}}, [6]uint64{1, 2, 3, 4, 5, 6}, TrainingRandomLimits{TrainingSampleLimits{3, 6}, 20}, 1, true); err == nil {
			t.Fatal("malformed trainer admitted")
		}
	}
	r := requireRunner(t)
	for name, p := range r.trainer.params {
		r.trainer.Moments[name] = NamedAdamState{Name: name, M: make([]float32, len(p)), V: nil}
		break
	}
	if err := r.validateUpdated(); err == nil {
		t.Fatal("malformed moments accepted")
	}
}
func TestTrainingRunnerOwnsRowsAndAlternatingShapes(t *testing.T) {
	base := requireRunner(t)
	a := PreparedTrainingRow{Batch: tinyFlowLMBatch(), ValidFrames: 2}
	b := a
	b.Batch.Frames = 2
	b.Batch.NormalizedLatents = append([]float32(nil), a.Batch.NormalizedLatents[:4]...)
	r, err := NewTrainingRunner(base.trainer, runIdentityFixture(), []PreparedTrainingRow{a, b}, [6]uint64{1, 2, 3, 4, 5, 6}, TrainingRandomLimits{TrainingSampleLimits{3, 6}, 20}, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	a.Batch.NormalizedLatents[0] = float32(math.NaN())
	for i := 0; i < 5; i++ {
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if r.workspace.frames != r.rows[r.run.core.order[r.run.core.cursor-1]].Batch.Frames {
		t.Fatal("workspace shape not current")
	}
}
func TestTrainingRunnerRestoreRejectedStateIsAtomic(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*TrainingRunner)
	}{
		{"optimizer", func(r *TrainingRunner) { r.trainer.Config.LearningRate *= 2 }},
		{"ema", func(r *TrainingRunner) { r.trainer.EMADecay = .8 }},
		{"negative-second-moment", func(r *TrainingRunner) {
			for _, name := range r.trainer.names {
				r.trainer.Moments[name].V[0] = -1
				break
			}
		}},
		{"buffer-shape", func(r *TrainingRunner) { r.trainer.FlowLM.LatentMean = append(r.trainer.FlowLM.LatentMean, 0) }},
		{"row-count", func(r *TrainingRunner) { r.run.core.rows = 2; r.run.core.order = []uint32{0, 1} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := requireRunner(t)
			if _, err := source.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			tc.change(source)
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := source.Checkpoint(dir, "incompatible", limits); err != nil {
				t.Fatal("fixture must be decoder-valid and coherently published", err)
			}
			fresh := requireRunner(t)
			before := mustFullState(t, fresh.trainer)
			beforeRun := snapshotRunFixture(t, fresh.run)
			if err := fresh.Restore(dir, limits); err == nil {
				t.Fatal("incompatible state restored")
			}
			if !reflect.DeepEqual(before, mustFullState(t, fresh.trainer)) || !bytes.Equal(beforeRun, snapshotRunFixture(t, fresh.run)) {
				t.Fatal("restore rejection mutated parameters/buffers/moments/EMA/run")
			}
			if _, err := fresh.Step(context.Background()); err != nil {
				t.Fatal("rejection damaged fresh runner", err)
			}
		})
	}
}
func TestTrainingRunnerHybridIdentityResume(t *testing.T) {
	for _, name := range []string{"ptx-attention-hybrid-v1", "vulkan-attention-hybrid-v1"} {
		create := func() *TrainingRunner {
			base := requireRunner(t)
			base.trainer.FlowLM.AttentionBackwardBackend = namedAttentionBackend{attentionBackendFunc(referenceAttentionBackend), name}
			id := runIdentityFixture()
			id.Backend = name
			r, err := NewTrainingRunner(base.trainer, id, base.rows, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
			if err != nil {
				t.Fatal(err)
			}
			return r
		}
		a := create()
		if _, err := a.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		limits := TrainingBundleLimits{1 << 20, 1 << 16}
		if _, err := a.Checkpoint(dir, "hybrid", limits); err != nil {
			t.Fatal(err)
		}
		b := create()
		if err := b.Restore(dir, limits); err != nil {
			t.Fatal(err)
		}
		if err := requireRunner(t).Restore(dir, limits); err == nil {
			t.Fatal("hybrid checkpoint restored into SIMD identity")
		}
		for _, r := range []*TrainingRunner{a, b} {
			if _, err := r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
		for name, p := range a.trainer.params {
			assertSliceClose(t, "hybrid resume "+name, b.trainer.params[name], p, 1e-6)
		}
	}
}
func TestTrainingRunnerCPUReplayRejectsHybrid(t *testing.T) {
	r := requireRunner(t)
	r.trainer.FlowLM.AttentionBackwardBackend = attentionBackendFunc(referenceAttentionBackend)
	if _, err := NewTrainingRunner(r.trainer, runIdentityFixture(), r.rows, [6]uint64{1, 2, 3, 4, 5, 6}, r.limits, 1, true); err == nil {
		t.Fatal("hybrid saved under SIMD replay identity")
	}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("changed backend executed under SIMD replay identity")
	}
	if r.trainer.StepCount != 0 {
		t.Fatal("invalid backend update advanced state")
	}
}
func TestTrainingRunnerGradientClipping(t *testing.T) {
	r := requireRunner(t)
	row := r.rows[0]
	samples := tinyTrainingStepSamples()
	_, g, err := PocketTrainingStep(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, row.Batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range r.trainer.bindings {
		clear(b.gradient(g))
	}
	values := r.trainer.bindings[0].gradient(g)
	if len(values) < 2 {
		t.Fatal("fixture gradient too small")
	}
	values[0], values[1] = 3, 4
	r.maxNorm = 2
	if err := r.clipGradients(g); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(values[0])-1.2) > 1e-6 || math.Abs(float64(values[1])-1.6) > 1e-6 {
		t.Fatal("independent3-4-5 clipping oracle", values[:2])
	}
	values[0], values[1] = math.MaxFloat32, math.MaxFloat32
	if err := r.clipGradients(g); err != nil {
		t.Fatal(err)
	}
	if math.Hypot(float64(values[0]), float64(values[1])) > 2.000001 {
		t.Fatal("large finite clipping")
	}
}
func TestTrainingRunnerPoisonRequiresRecovery(t *testing.T) {
	r := requireRunner(t)
	// Force an invalid optimiser state after constructor; no retry is safe
	// after an attempted update can have modified moments/parameters.
	for name, p := range r.trainer.params {
		r.trainer.Moments[name] = NamedAdamState{Name: name, M: make([]float32, len(p)), V: make([]float32, len(p))}
		r.trainer.Moments[name].M[0] = float32(math.Inf(1))
		break
	}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("invalid update succeeded")
	}
	if r.poison == nil {
		t.Fatal("uncertain update did not poison runner")
	}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("poisoned runner retried")
	}
	if _, err := r.Checkpoint(t.TempDir(), "bad", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
		t.Fatal("poisoned checkpoint")
	}
}
