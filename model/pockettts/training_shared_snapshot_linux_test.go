//go:build linux

package pockettts

import (
	"context"
	"errors"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// This fixture admits shared snapshot storage but deliberately cannot execute
// shared kernels. CPU updates initialise independent desired snapshot values.
// Unsupported mock device operations fail explicitly rather than emulate them.
type sharedSnapshotOnlyDriver struct{ hostResidentGraphDriver }

func (d *sharedSnapshotOnlyDriver) ValidateSharedInventory(context.Context, []train.ResidentBuffer, []int) error {
	return nil
}
func (d *sharedSnapshotOnlyDriver) Prefix(context.Context, train.ResidentBuffer, int, *train.SharedResidentAccess) (train.ResidentBuffer, error) {
	return nil, errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateSharedAccumulate(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, float32) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateCopy(context.Context, train.ResidentBuffer, train.ResidentBuffer, int) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) Copy(context.Context, train.ResidentBuffer, train.ResidentBuffer, int) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateNormStats(context.Context, train.ResidentBuffer, train.ResidentBuffer, int, bool) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateScaledUpdate(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, train.AdamWStep) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) UpdateScaled(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, train.AdamWStep) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateNormMerge(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, int64, uint32) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) NormMerge(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, int64, uint32) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) ValidateNormClip(context.Context, train.ResidentBuffer, train.ResidentBuffer, int64, float32) error {
	return errors.New("snapshot-only fixture")
}
func (d *sharedSnapshotOnlyDriver) NormClip(context.Context, train.ResidentBuffer, train.ResidentBuffer, int64, float32) error {
	return errors.New("snapshot-only fixture")
}

func TestPocketSharedSnapshotReuseCompletedLoad(t *testing.T) {
	for _, decay := range []float32{0, .9} {
		for _, completed := range []bool{false, true} {
			cpu, live := requireRunner(t), requireRunner(t)
			for _, r := range []*TrainingRunner{cpu, live} {
				r.trainer.EMADecay = decay
				if decay == 0 {
					r.trainer.EMA = nil
				}
			}
			if completed {
				for _, r := range []*TrainingRunner{cpu, live} {
					if _, err := r.Step(context.Background()); err != nil {
						t.Fatal(err)
					}
				}
			}
			b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", trainer: live.trainer, host: mustFullState(t, live.trainer)}
			before := b.host
			if _, err := cpu.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			desired := mustFullState(t, cpu.trainer)
			snapshot, err := sharedInitialSnapshot(desired)
			if err != nil {
				t.Fatal(err)
			}
			d := &sharedSnapshotOnlyDriver{}
			b.state, err = train.NewSharedWorkspaceResidentState(context.Background(), d, snapshot, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			allocations := d.allocations
			if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(desired, mustFullState(t, live.trainer)) || !reflect.DeepEqual(desired, b.host) || d.allocations != allocations {
				t.Fatal("reused completed snapshot parity")
			}
			for i, p := range before.Params {
				if &p.Values[0] != &b.host.Params[i].Values[0] || f32SlicesOverlap(p.Values, live.trainer.params[p.Name]) {
					t.Fatal("host/live P storage")
				}
			}
			if completed {
				for i, m := range before.Adam {
					if &m.M[0] != &b.host.Adam[i].M[0] || &m.V[0] != &b.host.Adam[i].V[0] {
						t.Fatal("host M/V not reused")
					}
				}
			}
			for i, e := range before.EMA {
				if &e.Values[0] != &b.host.EMA[i].Values[0] || f32SlicesOverlap(e.Values, live.trainer.EMA[e.Name]) {
					t.Fatal("host/live EMA storage")
				}
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestPocketSharedSnapshotStorageRejectsLiveAliases(t *testing.T) {
	for _, kind := range []string{"P", "M", "V", "EMA", "buffer", "cross", "empty"} {
		t.Run(kind, func(t *testing.T) {
			r, b := sharedCompletedCheckpointFixture(t)
			live := r.trainer
			name := b.host.Params[0].Name
			switch kind {
			case "P":
				b.host.Params[0].Values = live.params[name]
			case "M":
				b.host.Adam[0].M = live.Moments[b.host.Adam[0].Name].M
			case "V":
				b.host.Adam[0].V = live.Moments[b.host.Adam[0].Name].V
			case "EMA":
				b.host.EMA[0].Values = live.EMA[b.host.EMA[0].Name]
			case "buffer":
				b.host.Buffers[0].Values = fullBufferMap(live.FlowLM, live.Flow)[b.host.Buffers[0].Name]
			case "cross":
				b.host.Adam[0].M = b.host.Params[0].Values
			case "empty":
				b.host.Params[0].Values = nil
			}
			before := mustFullState(t, live)
			if err := validateSharedSnapshotStorage(b.host, live); err == nil {
				t.Fatal("unsafe source storage admitted")
			}
			if !reflect.DeepEqual(before, mustFullState(t, live)) {
				t.Fatal("storage admission mutated live trainer")
			}
		})
	}
	if validateSharedSnapshotStorage(FullTrainingState{}, nil) == nil {
		t.Fatal("nil trainer")
	}
}

func TestPocketSharedSnapshotReuseLateLoadFailure(t *testing.T) {
	cpu, live := requireRunner(t), requireRunner(t)
	before := mustFullState(t, live.trainer)
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	b.host, b.trainer = before, live.trainer
	for i := 0; i < 2; i++ {
		if _, err := cpu.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	// Completed native step2 cannot be copied into live step0. Validate the
	// rejection after all downloads, without treating the overwritten host
	// boundary as a coherent checkpoint or retrying it.
	desired := mustFullState(t, cpu.trainer)
	snapshot, err := sharedInitialSnapshot(desired)
	if err != nil {
		t.Fatal(err)
	}
	d := &sharedSnapshotOnlyDriver{}
	b.state, err = train.NewSharedWorkspaceResidentState(context.Background(), d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err == nil || b.terminal == nil || d.hostOptimizerDriver.downloads != 4*len(desired.Params) {
		t.Fatal("late live admission not terminal", err, d.hostOptimizerDriver.downloads)
	}
	// b.host may now contain downloaded values; compare a fresh independent
	// original CPU topology instead of the consumed/overwritten before arrays.
	original := requireRunner(t)
	if !reflect.DeepEqual(mustFullState(t, original.trainer), mustFullState(t, live.trainer)) {
		t.Fatal("late load failure wrote live state")
	}
	if b.host.Step != 0 || b.ValidateBoundary(live.trainer) == nil || b.Close() == nil {
		t.Fatal("partial host boundary committed/freed")
	}
	if _, err := b.sharedCheckpointState(live.trainer); err == nil {
		t.Fatal("partial boundary published")
	}
}

func TestPocketSharedSnapshotReuseDownloadFailure(t *testing.T) {
	cpu, live := requireRunner(t), requireRunner(t)
	before := mustFullState(t, live.trainer)
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	b.host, b.trainer = before, live.trainer
	if _, err := cpu.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	desired := mustFullState(t, cpu.trainer)
	snapshot, err := sharedInitialSnapshot(desired)
	if err != nil {
		t.Fatal(err)
	}
	d := &sharedSnapshotOnlyDriver{}
	b.state, err = train.NewSharedWorkspaceResidentState(context.Background(), d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.downloadFailure = true
	if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err == nil || b.terminal == nil {
		t.Fatal("download failure not terminal")
	}
	if !reflect.DeepEqual(before, mustFullState(t, live.trainer)) {
		t.Fatal("failed download mutated live trainer")
	}
	calls := d.hostOptimizerDriver.downloads
	if b.ValidateBoundary(live.trainer) == nil || b.Close() == nil {
		t.Fatal("failed boundary accepted/freed")
	}
	if _, err := b.sharedCheckpointState(live.trainer); err == nil {
		t.Fatal("failed boundary borrowed")
	}
	if b.loadSharedCompletedSnapshot(context.Background(), live.trainer) == nil || d.hostOptimizerDriver.downloads != calls {
		t.Fatal("failed snapshot retried native download")
	}
}
