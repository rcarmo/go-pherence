//go:build linux

package pockettts

import (
	"context"
	"errors"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func BenchmarkPocketSharedFirstMomentDestination(b *testing.B) {
	const n = 32768
	host := FullTrainingState{Version: 1, AdamW: DefaultAdamWConfig(), EMADecay: .9, Params: []NamedTrainingTensor{{Name: "p", Values: make([]float32, n)}}, Buffers: []NamedTrainingTensor{{Name: "fixed", Values: []float32{1}}}, EMA: []NamedTrainingTensor{{Name: "p", Values: make([]float32, n)}}}
	initial, err := sharedInitialSnapshot(host)
	if err != nil {
		b.Fatal(err)
	}
	backend := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: host}
	backend.rememberSharedInitialMoments(initial)
	for _, retained := range []bool{false, true} {
		name := "new-first-moments"
		if retained {
			name = "retained-first-moments"
		}
		b.Run(name, func(b *testing.B) {
			destination := host
			if retained {
				destination.Adam = backend.sharedInitialMoments
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := sharedInitialSnapshot(destination); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestPocketSharedInitialMomentsRetainedAndAdopted(t *testing.T) {
	for _, decay := range []float32{0, .9} {
		cpu, live := requireRunner(t), requireRunner(t)
		for _, r := range []*TrainingRunner{cpu, live} {
			r.trainer.EMADecay = decay
			if decay == 0 {
				r.trainer.EMA = nil
			}
		}
		b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: mustFullState(t, live.trainer), trainer: live.trainer}
		initial, err := sharedInitialSnapshot(b.host)
		if err != nil {
			t.Fatal(err)
		}
		b.rememberSharedInitialMoments(initial)
		if len(b.sharedInitialMoments) != len(b.host.Params) || len(b.host.Adam) != 0 || !sameResidentLiveBoundary(b.host, live.trainer) {
			t.Fatal("step-zero semantics changed")
		}
		for _, m := range b.sharedInitialMoments {
			p := initial.Parameters[m.Name]
			if &m.M[0] != &p.M[0] || &m.V[0] != &p.V[0] {
				t.Fatal("initial upload moments cloned")
			}
		}
		retained := b.sharedInitialMoments
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
		if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err != nil {
			t.Fatal(err)
		}
		if b.sharedInitialMoments != nil || !reflect.DeepEqual(desired, b.host) || !reflect.DeepEqual(desired, mustFullState(t, live.trainer)) {
			t.Fatal("completed moment adoption parity")
		}
		for i, m := range retained {
			if &m.M[0] != &b.host.Adam[i].M[0] || &m.V[0] != &b.host.Adam[i].V[0] {
				t.Fatal("retained moments not adopted")
			}
		}
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPocketSharedInitialMomentsDestinationAllocationBound(t *testing.T) {
	r := requireRunner(t)
	host := mustFullState(t, r.trainer)
	initial, err := sharedInitialSnapshot(host)
	if err != nil {
		t.Fatal(err)
	}
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: host}
	b.rememberSharedInitialMoments(initial)
	destination := host
	destination.Adam = b.sharedInitialMoments
	prior := testing.AllocsPerRun(10, func() {
		if _, err := sharedInitialSnapshot(host); err != nil {
			panic(err)
		}
	})
	retained := testing.AllocsPerRun(10, func() {
		if _, err := sharedInitialSnapshot(destination); err != nil {
			panic(err)
		}
	})
	if retained >= prior {
		t.Fatal("retained first moment destination allocation regression", retained, prior)
	}
	t.Logf("tiny destination allocations: new=%g retained=%g; all parameter arrays stay independently owned", prior, retained)
}

func TestPocketSharedInitialMomentsPrepareLifecycle(t *testing.T) {
	r := requireRunner(t)
	b, _ := installResidentGraph(t, r.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if !b.active || b.state == nil || len(b.sharedInitialMoments) != len(b.host.Params) || len(b.host.Adam) != 0 || !sameResidentLiveBoundary(b.host, r.trainer) {
		t.Fatal("actual prepare lost initial moments/host guard")
	}
	calls := d.allocations
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil || d.allocations != calls {
		t.Fatal("active prepare duplicated storage")
	}
	if b.Close() == nil {
		t.Fatal("active owner closed")
	}
	// No graph/update occurred. Return this clean fixture to an idle boundary
	// solely to check initial checkpoint borrowing keeps its empty-Adam shape.
	b.active = false
	state, err := b.sharedCheckpointState(r.trainer)
	if err != nil || len(state.Adam) != 0 {
		t.Fatal("initial checkpoint contract", err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

type sharedInitialUploadFailureBuffer struct {
	train.ResidentBuffer
	driver *sharedInitialUploadFailureDriver
}

func (b *sharedInitialUploadFailureBuffer) Upload(context.Context, []float32) error {
	b.driver.uploadAttempts++
	return errors.New("initial upload uncertain")
}

type sharedInitialUploadFailureDriver struct {
	sharedSnapshotOnlyDriver
	uploadAttempts int
}

func (d *sharedInitialUploadFailureDriver) NewBuffer(ctx context.Context, n int) (train.ResidentBuffer, error) {
	b, err := d.hostOptimizerDriver.NewBuffer(ctx, n)
	if b == nil {
		return nil, err
	}
	return &sharedInitialUploadFailureBuffer{b, d}, err
}

func TestPocketSharedInitialMomentsFailedPrepareRetainsOwner(t *testing.T) {
	r := requireRunner(t)
	b, _ := installResidentGraph(t, r.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	d := &sharedInitialUploadFailureDriver{}
	b.Driver = d
	before := mustFullState(t, r.trainer)
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err == nil || b.state == nil || b.terminal == nil || d.uploadAttempts != 1 {
		t.Fatal("failed initial upload did not retain terminal owner")
	}
	if b.sharedInitialMoments != nil || b.host.Params != nil || !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
		t.Fatal("failed constructor adopted host storage or changed live")
	}
	calls := d.uploadAttempts
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil || b.Close() == nil || d.uploadAttempts != calls {
		t.Fatal("failed constructor retried/freed")
	}
}

func TestPocketSharedInitialMomentsAdmissionAndFailure(t *testing.T) {
	for _, kind := range []string{"count", "alias", "download"} {
		t.Run(kind, func(t *testing.T) {
			cpu, live := requireRunner(t), requireRunner(t)
			b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: mustFullState(t, live.trainer), trainer: live.trainer}
			before := mustFullState(t, live.trainer)
			initial, err := sharedInitialSnapshot(b.host)
			if err != nil {
				t.Fatal(err)
			}
			b.rememberSharedInitialMoments(initial)
			if _, err := cpu.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			snapshot, err := sharedInitialSnapshot(mustFullState(t, cpu.trainer))
			if err != nil {
				t.Fatal(err)
			}
			d := &sharedSnapshotOnlyDriver{}
			b.state, err = train.NewSharedWorkspaceResidentState(context.Background(), d, snapshot, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "count":
				b.sharedInitialMoments = b.sharedInitialMoments[:1]
			case "alias":
				b.sharedInitialMoments[0].M = b.host.Params[0].Values
			case "download":
				d.downloadFailure = true
			}
			if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err == nil || b.terminal == nil {
				t.Fatal("invalid initial moments accepted")
			}
			if len(b.host.Adam) != 0 || b.sharedInitialMoments == nil || !reflect.DeepEqual(before, mustFullState(t, live.trainer)) {
				t.Fatal("failed moment adoption mutated live/metadata")
			}
			if kind != "download" && d.hostOptimizerDriver.downloads != 0 {
				t.Fatal("bad destination downloaded")
			}
			if b.Close() == nil {
				t.Fatal("uncertain owner freed")
			}
		})
	}
	for _, b := range []*ResidentTrainingBackend{{Scope: "vulkan-resident-devicehandoff-hybrid-v1"}, {Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: FullTrainingState{Step: 1}}, {Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: FullTrainingState{Adam: []NamedAdamState{{Name: "existing"}}}}} {
		b.rememberSharedInitialMoments(train.ResidentSnapshot{})
		if b.sharedInitialMoments != nil {
			t.Fatal("legacy/restored moments replaced")
		}
	}
}
