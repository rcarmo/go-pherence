//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"errors"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

type hostOptimizerBuffer struct {
	values []float32
	driver *hostOptimizerDriver
}

func (b *hostOptimizerBuffer) Upload(ctx context.Context, x []float32) error {
	b.driver.uploads++
	copy(b.values, x)
	return ctx.Err()
}
func (b *hostOptimizerBuffer) Download(ctx context.Context, x []float32) error {
	b.driver.downloads++
	if b.driver.downloadFailure {
		return errors.New("download failed")
	}
	copy(x, b.values)
	return ctx.Err()
}
func (b *hostOptimizerBuffer) Close() error { return nil }

type hostOptimizerDriver struct {
	allocations, updates, uploads, downloads int
	failAt                                   int
	downloadFailure                          bool
}

func (d *hostOptimizerDriver) NewBuffer(ctx context.Context, n int) (train.ResidentBuffer, error) {
	d.allocations++
	return &hostOptimizerBuffer{values: make([]float32, n), driver: d}, nil
}
func (d *hostOptimizerDriver) Accumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	x, y, z := a.(*hostOptimizerBuffer).values, b.(*hostOptimizerBuffer).values, dst.(*hostOptimizerBuffer).values
	for i := range z {
		z[i] = x[i] + y[i]*scale
	}
	return ctx.Err()
}
func (d *hostOptimizerDriver) NormStats(ctx context.Context, dst, x train.ResidentBuffer, n int, positive bool) error {
	v, out := x.(*hostOptimizerBuffer).values, dst.(*hostOptimizerBuffer).values
	for chunk := 0; chunk < len(out)/3; chunk++ {
		begin, end := chunk*256, min(n, (chunk+1)*256)
		maximum, flag := float32(0), float32(0)
		for _, v := range v[begin:end] {
			if !isFinite(v) || (positive && v < 0) {
				flag = 1
			} else {
				maximum = max(maximum, float32(math.Abs(float64(v))))
			}
		}
		var sum, correction float32
		if maximum > 0 {
			for _, v := range v[begin:end] {
				a := v / maximum
				term := a*a - correction
				next := sum + term
				correction = (next - sum) - term
				sum = next
			}
		}
		out[3*chunk], out[3*chunk+1], out[3*chunk+2] = maximum, sum, flag
	}
	return ctx.Err()
}
func (d *hostOptimizerDriver) ValidateUpdate(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, train.AdamWStep) error {
	return nil
}
func (d *hostOptimizerDriver) Update(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, s train.AdamWStep) error {
	d.updates++
	if d.failAt == d.updates {
		return errors.New("native update failed")
	}
	pv, gv, mv, vv, ev := p.(*hostOptimizerBuffer).values, g.(*hostOptimizerBuffer).values, m.(*hostOptimizerBuffer).values, v.(*hostOptimizerBuffer).values, ema.(*hostOptimizerBuffer).values
	for i := range pv {
		x := gv[i] * s.GradientScale
		mv[i] = s.Beta1*mv[i] + (1-s.Beta1)*x
		vv[i] = s.Beta2*vv[i] + ((1-s.Beta2)*x)*x
		pv[i] *= 1 - s.LearningRate*s.WeightDecay
		pv[i] -= s.LearningRate * (mv[i] / s.Bias1) / (float32(math.Sqrt(float64(vv[i]/s.Bias2))) + s.Epsilon)
		if s.EMADecay > 0 {
			ev[i] = s.EMADecay*ev[i] + (1-s.EMADecay)*pv[i]
		}
	}
	return ctx.Err()
}
func optimizerRunner(t *testing.T, scope string) (*TrainingRunner, *ResidentOptimizerBackend, *hostOptimizerDriver) {
	t.Helper()
	base := requireRunner(t)
	d := &hostOptimizerDriver{}
	backend := &ResidentOptimizerBackend{Driver: d, Scope: scope, MaxBytes: 1 << 20}
	base.trainer.OptimizerBackend = backend
	id := runIdentityFixture()
	id.Backend = scope
	r, err := NewTrainingRunner(base.trainer, id, base.rows, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	return r, backend, d
}
func TestTrainingOptimizerRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-optimizer-hybrid-v1", "vulkan-optimizer-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			a, backend, d := optimizerRunner(t, scope)
			cpu := requireRunner(t)
			if backend.state != nil || d.allocations != 0 {
				t.Fatal("native state before restore")
			}
			ma, err := a.Step(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			mc, err := cpu.Step(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			assertClose64(t, "optimizer hybrid loss", ma.Loss, mc.Loss, 3e-5)
			for key, p := range cpu.trainer.params {
				assertSliceClose(t, "optimizer CPU reference "+key, a.trainer.params[key], p, 4e-6)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "optimizer", limits); err != nil {
				t.Fatal(err)
			}
			b, fresh, bd := optimizerRunner(t, scope)
			if err := b.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			if fresh.state != nil || bd.allocations != 0 {
				t.Fatal("restore initialised stale native state")
			}
			beforeAlloc := d.allocations
			for _, r := range []*TrainingRunner{a, b} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for key, p := range a.trainer.params {
				assertSliceClose(t, "optimizer resume "+key, b.trainer.params[key], p, 1e-6)
				assertSliceClose(t, "optimizer M "+key, b.trainer.Moments[key].M, a.trainer.Moments[key].M, 1e-6)
				assertSliceClose(t, "optimizer V "+key, b.trainer.Moments[key].V, a.trainer.Moments[key].V, 1e-6)
				assertSliceClose(t, "optimizer EMA "+key, b.trainer.EMA[key], a.trainer.EMA[key], 1e-6)
			}
			if d.allocations != beforeAlloc {
				t.Fatal("steady state reallocated resident buffers")
			}
			if !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, b.run)) {
				t.Fatal("raw replay differs")
			}
			for _, other := range []*TrainingRunner{requireRunner(t), flowRunner(t, "ptx-flow-hybrid-v1"), transformerRunner(t, "ptx-transformer-hybrid-v1")} {
				before := mustFullState(t, other.trainer)
				if err := other.Restore(dir, limits); err == nil {
					t.Fatal("cross scope restored")
				}
				if !reflect.DeepEqual(before, mustFullState(t, other.trainer)) {
					t.Fatal("restore rejection mutated")
				}
			}
			if backend.Close() != nil || fresh.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingOptimizerFailureStopping(t *testing.T) {
	for _, failAt := range []int{1, 2} {
		r, backend, d := optimizerRunner(t, "ptx-optimizer-hybrid-v1")
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		d.failAt = failAt
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("failed optimiser accepted")
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed native update mutated host/replay")
		}
		updates := d.updates
		if _, err := r.Step(context.Background()); err == nil || d.updates != updates {
			t.Fatal("failed optimiser retried")
		}
		if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
			t.Fatal("failed optimiser checkpointed")
		}
		if backend.Close() == nil {
			t.Fatal("uncertain owner closed")
		}
	}
	r, backend, _ := optimizerRunner(t, "ptx-optimizer-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	r.trainer.params[r.trainer.names[0]][0] += 1
	before := mustFullState(t, r.trainer)
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("host mutation admitted")
	}
	if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
		t.Fatal("rejection mutated host")
	}
	backend.Scope = "vulkan-optimizer-hybrid-v1"
	if _, err := r.Checkpoint(t.TempDir(), "changed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
		t.Fatal("changed scope checkpointed")
	}
}
func TestTrainingOptimizerBackendAdmission(t *testing.T) {
	var nilBackend *ResidentOptimizerBackend
	if nilBackend.Close() != nil || nilBackend.TrainingOptimizerBackendName() != "unknown" {
		t.Fatal("nil backend")
	}
	r, backend, _ := optimizerRunner(t, "ptx-optimizer-hybrid-v1")
	r.trainer.Flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("unbound combined scope")
	}
	if backend.UpdateTrainer(nil, r.trainer, TrainingStepGradients{}, 1) == nil {
		t.Fatal("nil context")
	}
	backend.Scope = "unknown"
	if backend.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 1) == nil {
		t.Fatal("unknown scope")
	}
}

func TestTrainingOptimizerCheckpointBoundary(t *testing.T) {
	r, backend, _ := optimizerRunner(t, "ptx-optimizer-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := backend.ValidateBoundary(r.trainer); err != nil {
		t.Fatal(err)
	}
	r.trainer.Moments[r.trainer.names[0]].M[0] += 1
	if _, err := r.Checkpoint(t.TempDir(), "mutated", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
		t.Fatal("checkpoint accepted diverged host moments")
	}
	if err := backend.Close(); err != nil {
		t.Fatal(err)
	}
	if err := backend.ValidateBoundary(r.trainer); err == nil {
		t.Fatal("closed optimiser boundary")
	}
	if backend.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 1) == nil {
		t.Fatal("closed optimiser updated")
	}
}

func TestTrainingOptimizerMalformedGradientAdmission(t *testing.T) {
	r, b, d := optimizerRunner(t, "ptx-optimizer-hybrid-v1")
	if b.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 1) == nil || d.allocations != 0 {
		t.Fatal("malformed gradients reached allocation")
	}
}
