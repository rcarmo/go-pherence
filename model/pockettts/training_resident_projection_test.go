//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"reflect"
	"testing"
)

func TestTrainingResidentProjectionRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-projection-hybrid-v1", "vulkan-resident-projection-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			a, b, d := residentGraphRunner(t, scope)
			if _, err := a.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "projection", limits); err != nil {
				t.Fatal(err)
			}
			resumed, rb, _ := residentGraphRunner(t, scope)
			if err := resumed.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			beforeAlloc := d.allocations
			for _, r := range []*TrainingRunner{a, resumed} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for name, p := range a.trainer.params {
				assertSliceClose(t, "projection resume "+name, resumed.trainer.params[name], p, 1e-6)
				assertSliceClose(t, "projection resume M "+name, resumed.trainer.Moments[name].M, a.trainer.Moments[name].M, 1e-6)
				assertSliceClose(t, "projection resume V "+name, resumed.trainer.Moments[name].V, a.trainer.Moments[name].V, 1e-6)
				assertSliceClose(t, "projection resume EMA "+name, resumed.trainer.EMA[name], a.trainer.EMA[name], 1e-6)
			}
			if beforeAlloc != d.allocations || !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, resumed.run)) {
				t.Fatal("resident allocations/replay")
			}
			legacy, _, _ := residentGraphRunner(t, "ptx-resident-weight-hybrid-v1")
			before := mustFullState(t, legacy.trainer)
			if err := legacy.Restore(dir, limits); err == nil {
				t.Fatal("projection resumed under old weight scope")
			}
			if !reflect.DeepEqual(before, mustFullState(t, legacy.trainer)) {
				t.Fatal("restore rejection mutated")
			}
			if b.Close() != nil || rb.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentProjectionEmptyVoice(t *testing.T) {
	base := requireRunner(t)
	row := base.rows[0]
	row.Batch.VoiceFrames = 0
	row.Batch.VoiceLatents = nil
	b, d := installResidentGraph(t, base.trainer, "vulkan-resident-projection-hybrid-v1")
	id := runIdentityFixture()
	id.Backend = b.Scope
	r, err := NewTrainingRunner(base.trainer, id, []PreparedTrainingRow{row}, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for i, name := range r.trainer.names {
		if name == "flow_lm.speaker_proj_weight" {
			for _, v := range d.validatedGradients[i] {
				if v != 0 {
					t.Fatal("empty voice projection received gradient")
				}
			}
		}
	}
}
func TestTrainingResidentProjectionFailures(t *testing.T) {
	// Weight forward, speaker/input forward, EOS forward/backward, speaker/input backward.
	for _, at := range []int{2, 3, 5, 19, 21, 22} {
		r, b, d := residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
		d.failTape = at
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("projection failure admitted", at)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed projection changed host/replay")
		}
		calls := d.tapes
		if _, err := r.Step(context.Background()); err == nil || calls != d.tapes {
			t.Fatal("failed projection retried")
		}
		if b.Close() == nil {
			t.Fatal("uncertain owner released")
		}
	}
	r, b, _ := residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
	r.trainer.FlowLM.ProjectionBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("projection removed under scope")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err == nil {
		t.Fatal("wrong owner")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-weight-hybrid-v1")
	legacy.trainer.FlowLM.ProjectionBackend = lb
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("projection injected under legacy scope")
	}
}
func TestTrainingResidentProjectionAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
	if _, err := b.EvaluateProjection(r.trainer.FlowLM, "speaker", nil, nil, 1, false); err == nil {
		t.Fatal("inactive")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateProjection(r.trainer.FlowLM, "unknown", nil, nil, 1, false); err == nil || b.terminal == nil {
		t.Fatal("unknown kind")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateProjection(tinyFlowLMTraining(), "eos", nil, nil, 1, false); err == nil {
		t.Fatal("wrong model")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-projection-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateProjection(r.trainer.FlowLM, "speaker-empty", nil, nil, 0, false); err == nil {
		t.Fatal("implicit empty gradient")
	}
}
