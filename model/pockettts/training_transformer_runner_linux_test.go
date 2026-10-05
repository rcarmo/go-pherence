//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"testing"
)

type namedHostTransformerExecutor struct {
	*hostTransformerTapeExecutor
	name string
}

func (e namedHostTransformerExecutor) TrainingTransformerBackendName() string { return e.name }
func transformerRunner(t *testing.T, name string) *TrainingRunner {
	t.Helper()
	base := requireRunner(t)
	base.trainer.FlowLM.TransformerBackend = &TransformerTapeTrainingBackend{Executor: namedHostTransformerExecutor{&hostTransformerTapeExecutor{t: t}, name}, MaxBytes: 64 << 20}
	id := runIdentityFixture()
	id.Backend = name
	r, err := NewTrainingRunner(base.trainer, id, base.rows, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func TestTrainingTransformerRunnerResume(t *testing.T) {
	for _, name := range []string{"ptx-transformer-hybrid-v1", "vulkan-transformer-hybrid-v1"} {
		t.Run(name, func(t *testing.T) {
			a := transformerRunner(t, name)
			if _, err := a.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "transformer", limits); err != nil {
				t.Fatal(err)
			}
			b := transformerRunner(t, name)
			if err := b.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			for _, r := range []*TrainingRunner{a, b} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for key, p := range a.trainer.params {
				assertSliceClose(t, "transformer resume "+key, b.trainer.params[key], p, 1e-6)
				assertSliceClose(t, "transformer resume EMA "+key, b.trainer.EMA[key], a.trainer.EMA[key], 1e-6)
			}
			if !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, b.run)) {
				t.Fatal("raw replay differs")
			}
			for _, fresh := range []*TrainingRunner{requireRunner(t), flowRunner(t, "ptx-flow-hybrid-v1"), flowRunner(t, "vulkan-flow-hybrid-v1")} {
				before := mustFullState(t, fresh.trainer)
				if err := fresh.Restore(dir, limits); err == nil {
					t.Fatal("cross scope restored")
				}
				if !reflect.DeepEqual(before, mustFullState(t, fresh.trainer)) {
					t.Fatal("restore rejection mutated")
				}
			}
			r := transformerRunner(t, name)
			r.trainer.FlowLM.TransformerBackend = nil
			if _, err := r.Step(context.Background()); err == nil {
				t.Fatal("changed scope executed")
			}
			if _, err := r.Checkpoint(dir, "changed", limits); err == nil {
				t.Fatal("changed scope checkpointed")
			}
			if err := r.Restore(dir, limits); err == nil {
				t.Fatal("changed scope restored")
			}
		})
	}
}
func TestTrainingTransformerRunnerFailure(t *testing.T) {
	for _, failAt := range []int{1, 2} {
		r := transformerRunner(t, "ptx-transformer-hybrid-v1")
		sentinel := errors.New("transformer failed")
		e := &hostTransformerTapeExecutor{t: t, failAt: failAt, err: sentinel}
		r.trainer.FlowLM.TransformerBackend = &TransformerTapeTrainingBackend{Executor: namedHostTransformerExecutor{e, "ptx-transformer-hybrid-v1"}, MaxBytes: 64 << 20}
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); !errors.Is(err, sentinel) {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed step mutated params/optimizer/replay")
		}
		if _, err := r.Step(context.Background()); err == nil || e.calls != failAt {
			t.Fatal("failed transformer retried")
		}
		if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
			t.Fatal("failed transformer checkpointed")
		}
	}
	r := transformerRunner(t, "ptx-transformer-hybrid-v1")
	r.trainer.Flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("unbound combined flow+transformer")
	}
}
