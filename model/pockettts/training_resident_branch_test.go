//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"testing"
)

func TestTrainingResidentBranchRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-branch-hybrid-v1", "vulkan-resident-branch-hybrid-v1"} {
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
			legacy, _, _ := residentGraphRunner(t, "ptx-resident-loss-hybrid-v1")
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
func TestTrainingResidentBranchFailures(t *testing.T) {
	baseline, _, driver := residentGraphRunner(t, "ptx-resident-branch-hybrid-v1")
	if _, err := baseline.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= driver.tapes; at++ {
		r, b, d := residentGraphRunner(t, "ptx-resident-branch-hybrid-v1")
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
	r, b, _ := residentGraphRunner(t, "ptx-resident-branch-hybrid-v1")
	r.trainer.Flow.BranchBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("removed owner")
	}
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil {
		t.Fatal("admission owner")
	}
	legacy, _, _ := residentGraphRunner(t, "ptx-resident-loss-hybrid-v1")
	legacy.trainer.Flow.TrainingBackend = b
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("legacy injection")
	}
}
func TestTrainingResidentBranchObjectiveWeights(t *testing.T) {
	for _, weight := range []float32{0, .1, 1} {
		for _, p := range []float32{0, .75, 1} {
			t.Run(fmt.Sprintf("eos%g-flow%g", weight, p), func(t *testing.T) {
				testResidentGraphFullStepConfig(t, "vulkan-resident-branch-hybrid-v1", TrainingStepConfig{PEqual: p, EOSLossWeight: weight})
			})
		}
	}
}
func TestTrainingResidentBranchAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-branch-hybrid-v1")
	if _, err := b.EvaluateBranch(r.trainer.Flow, "scale", [][]float32{{1}}, 1); err == nil {
		t.Fatal("inactive")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateBranch(&FlowHeadCPU{}, "scale", [][]float32{{1}}, 1); err == nil {
		t.Fatal("owner")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-branch-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	b.MaxTapeBytes = 1
	if _, err := b.EvaluateBranch(r.trainer.Flow, "scale", [][]float32{{1}}, 1); err == nil || b.terminal == nil {
		t.Fatal("budget")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-loss-hybrid-v1")
	legacy.trainer.Flow.BranchBackend = lb
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("legacy injection")
	}
}
