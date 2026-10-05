//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type namedHostFlowExecutor struct {
	hostFlowTapeExecutor
	name   string
	calls  *int
	failAt int
	err    error
}

func (e namedHostFlowExecutor) TrainingFlowBackendName() string { return e.name }
func (e namedHostFlowExecutor) ExecuteFlowTape(tape FlowDualTrainingTape) (FlowDualTrainingResult, error) {
	if e.calls != nil {
		*e.calls++
		if *e.calls == e.failAt {
			return FlowDualTrainingResult{}, e.err
		}
	}
	return e.hostFlowTapeExecutor.ExecuteFlowTape(tape)
}
func flowRunner(t *testing.T, name string) *TrainingRunner {
	t.Helper()
	base := requireRunner(t)
	base.trainer.Flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: namedHostFlowExecutor{hostFlowTapeExecutor: hostFlowTapeExecutor{t}, name: name}, MaxBytes: 64 << 20}
	id := runIdentityFixture()
	id.Backend = name
	r, err := NewTrainingRunner(base.trainer, id, base.rows, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func TestTrainingFlowRunnerHybridResume(t *testing.T) {
	for _, name := range []string{"ptx-flow-hybrid-v1", "vulkan-flow-hybrid-v1"} {
		t.Run(name, func(t *testing.T) {
			a := flowRunner(t, name)
			cpu := requireRunner(t)
			ma, err := a.Step(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			mc, err := cpu.Step(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			assertClose64(t, "host flow hybrid vs CPU loss", ma.Loss, mc.Loss, 3e-5)
			for key, p := range cpu.trainer.params {
				assertSliceClose(t, "host flow update vs CPU "+key, a.trainer.params[key], p, 5e-5)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "flow", limits); err != nil {
				t.Fatal(err)
			}
			b := flowRunner(t, name)
			if err := b.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			for _, r := range []*TrainingRunner{a, b} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for key, p := range a.trainer.params {
				assertSliceClose(t, "flow resume "+key, b.trainer.params[key], p, 1e-6)
				assertSliceClose(t, "flow resume M "+key, b.trainer.Moments[key].M, a.trainer.Moments[key].M, 1e-6)
				assertSliceClose(t, "flow resume V "+key, b.trainer.Moments[key].V, a.trainer.Moments[key].V, 1e-6)
				assertSliceClose(t, "flow resume EMA "+key, b.trainer.EMA[key], a.trainer.EMA[key], 1e-6)
			}
			if !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, b.run)) {
				t.Fatal("raw replay differs")
			}
			for _, other := range []string{"simd", "ptx-attention-hybrid-v1", "vulkan-attention-hybrid-v1", "ptx-flow-hybrid-v1", "vulkan-flow-hybrid-v1"} {
				if other == name {
					continue
				}
				fresh := requireRunner(t)
				if other != "simd" {
					id := runIdentityFixture()
					id.Backend = other
					if other == "ptx-flow-hybrid-v1" || other == "vulkan-flow-hybrid-v1" {
						fresh = flowRunner(t, other)
					} else {
						fresh.trainer.FlowLM.AttentionBackwardBackend = namedAttentionBackend{attentionBackendFunc(referenceAttentionBackend), other}
						fresh, err = NewTrainingRunner(fresh.trainer, id, fresh.rows, [6]uint64{1, 2, 3, 4, 5, 6}, fresh.limits, 1, true)
						if err != nil {
							t.Fatal(err)
						}
					}
				}
				before := mustFullState(t, fresh.trainer)
				beforeRun := snapshotRunFixture(t, fresh.run)
				if err := fresh.Restore(dir, limits); err == nil {
					t.Fatal("cross scope resume accepted", other)
				}
				if !reflect.DeepEqual(before, mustFullState(t, fresh.trainer)) || !bytes.Equal(beforeRun, snapshotRunFixture(t, fresh.run)) {
					t.Fatal("cross scope rejection mutated state")
				}
			}
			pinChanged := flowRunner(t, name)
			pinChanged.run.core.identity.OptionsSHA256 = strings.Repeat("9", 64)
			if err := pinChanged.Restore(dir, limits); err == nil {
				t.Fatal("changed runtime/options identity admitted")
			}
		})
	}
}
func TestTrainingFlowRunnerScopeChanges(t *testing.T) {
	for _, scope := range []string{"simd", "ptx-flow-hybrid-v1", "vulkan-flow-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			r := requireRunner(t)
			if scope != "simd" {
				r = flowRunner(t, scope)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := r.Checkpoint(dir, "initial", limits); err != nil {
				t.Fatal(err)
			}
			if scope == "simd" {
				r.trainer.Flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: namedHostFlowExecutor{hostFlowTapeExecutor: hostFlowTapeExecutor{t}, name: "ptx-flow-hybrid-v1"}, MaxBytes: 64 << 20}
			} else {
				r.trainer.Flow.TrainingBackend = nil
			}
			before := mustFullState(t, r.trainer)
			raw := snapshotRunFixture(t, r.run)
			if _, err := NewTrainingRunner(r.trainer, r.run.core.identity, r.rows, [6]uint64{1, 2, 3, 4, 5, 6}, r.limits, 1, true); err == nil {
				t.Fatal("mismatched construction")
			}
			if _, err := r.Step(context.Background()); err == nil {
				t.Fatal("changed scope executed")
			}
			if _, err := r.Checkpoint(dir, "changed", limits); err == nil {
				t.Fatal("changed scope checkpointed")
			}
			if err := r.Restore(dir, limits); err == nil {
				t.Fatal("changed scope restored")
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
				t.Fatal("rejection mutated state")
			}
		})
	}
	r := flowRunner(t, "ptx-flow-hybrid-v1")
	r.trainer.FlowLM.AttentionBackwardBackend = namedAttentionBackend{attentionBackendFunc(referenceAttentionBackend), "ptx-attention-hybrid-v1"}
	if _, err := NewTrainingRunner(r.trainer, r.run.core.identity, r.rows, [6]uint64{1, 2, 3, 4, 5, 6}, r.limits, 1, true); err == nil {
		t.Fatal("unbound combined scope")
	}
}
func TestTrainingFlowRunnerFailedStepStops(t *testing.T) {
	for _, failAt := range []int{1, 3, 6} {
		t.Run(string(rune('0'+failAt)), func(t *testing.T) {
			r := flowRunner(t, "ptx-flow-hybrid-v1")
			calls := 0
			sentinel := errors.New("flow executor failed")
			r.trainer.Flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: namedHostFlowExecutor{hostFlowTapeExecutor: hostFlowTapeExecutor{t}, name: "ptx-flow-hybrid-v1", calls: &calls, failAt: failAt, err: sentinel}, MaxBytes: 64 << 20}
			before := mustFullState(t, r.trainer)
			raw := snapshotRunFixture(t, r.run)
			if _, err := r.Step(context.Background()); !errors.Is(err, sentinel) {
				t.Fatal(err)
			}
			if calls != failAt || r.poison == nil {
				t.Fatal("failed scope not stopped", calls)
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
				t.Fatal("failed backward changed model/optimizer/RNG")
			}
			if _, err := r.Step(context.Background()); err == nil || calls != failAt {
				t.Fatal("failed flow retried")
			}
			if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
				t.Fatal("failed flow checkpointed")
			}
		})
	}
}
