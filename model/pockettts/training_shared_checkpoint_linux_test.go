//go:build linux

package pockettts

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// The scalar resident constructor supplies a real completed StepCount for the
// checkpoint contract. This fixture does not execute the shared driver/tapes:
// the numerical update is the existing CPU runner, and snapshot provenance is
// its independent State() copy. No native arithmetic qualification follows.
func sharedCompletedCheckpointFixture(t *testing.T) (*TrainingRunner, *ResidentTrainingBackend) {
	t.Helper()
	r := requireRunner(t)
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, r.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	host := mustFullState(t, r.trainer)
	snapshot, err := sharedInitialSnapshot(host)
	if err != nil {
		t.Fatal(err)
	}
	state, err := train.NewResidentState(context.Background(), d, snapshot, b.MaxResidentBytes)
	if err != nil {
		t.Fatal(err)
	}
	b.state, b.host, b.trainer = state, host, r.trainer
	r.run.core.identity.Backend = b.Scope
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	return r, b
}

func TestPocketSharedCheckpointCompletedOwnership(t *testing.T) {
	r, b := sharedCompletedCheckpointFixture(t)
	borrowed, err := b.sharedCheckpointState(r.trainer)
	if err != nil || borrowed.Step != 1 || len(borrowed.Adam) != len(borrowed.Params) {
		t.Fatal("completed boundary", err)
	}
	liveBuffers := fullBufferMap(r.trainer.FlowLM, r.trainer.Flow)
	for _, p := range borrowed.Params {
		if f32SlicesOverlap(p.Values, r.trainer.params[p.Name]) {
			t.Fatal("borrowed live parameter", p.Name)
		}
	}
	for _, m := range borrowed.Adam {
		live := r.trainer.Moments[m.Name]
		if f32SlicesOverlap(m.M, live.M) || f32SlicesOverlap(m.V, live.V) {
			t.Fatal("borrowed live moments", m.Name)
		}
	}
	for _, e := range borrowed.EMA {
		if f32SlicesOverlap(e.Values, r.trainer.EMA[e.Name]) {
			t.Fatal("borrowed live EMA", e.Name)
		}
	}
	for _, fixed := range borrowed.Buffers {
		if f32SlicesOverlap(fixed.Values, liveBuffers[fixed.Name]) {
			t.Fatal("borrowed live frozen buffer", fixed.Name)
		}
	}
	before, runBefore := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := r.Checkpoint(dir, "completed", limits); err != nil {
		t.Fatal(err)
	}
	bundle, err := OpenTrainingBundle(dir, r.run.core.identity, limits)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadFullTrainingStateBinary(bundle.Model, bundle.ModelBytes)
	closeErr := bundle.Close()
	if err != nil || closeErr != nil || !reflect.DeepEqual(decoded, before) {
		t.Fatal("synchronous completed checkpoint", err, closeErr)
	}
	if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !reflect.DeepEqual(runBefore, snapshotRunFixture(t, r.run)) || !sameResidentLiveBoundary(b.host, r.trainer) {
		t.Fatal("serialisation changed completed boundary/replay")
	}
	// Public State still returns storage independent of the borrowed snapshot.
	public := mustFullState(t, r.trainer)
	public.Params[0].Values[0] += 1
	public.Adam[0].M[0] += 1
	public.EMA[0].Values[0] += 1
	if !reflect.DeepEqual(before, b.host) || !sameResidentLiveBoundary(b.host, r.trainer) {
		t.Fatal("public State ownership changed")
	}
}

func TestPocketSharedCheckpointWriterFailuresPreserveBoundary(t *testing.T) {
	for _, kind := range []string{"directory", "byte-limit", "existing-generation"} {
		t.Run(kind, func(t *testing.T) {
			r, b := sharedCompletedCheckpointFixture(t)
			before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			switch kind {
			case "directory":
				dir = filepath.Join(dir, "absent")
			case "byte-limit":
				limits.MaxModelBytes = trainingBundlePrefixBytes
			case "existing-generation":
				if err := os.WriteFile(filepath.Join(dir, "rejected.model.bin"), []byte("preserved generation"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := r.Checkpoint(dir, "rejected", limits); err == nil {
				t.Fatal("expected writer failure")
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !reflect.DeepEqual(before, b.host) || !reflect.DeepEqual(raw, snapshotRunFixture(t, r.run)) {
				t.Fatal("failed writer mutated model/replay/owned boundary")
			}
			if err := b.ValidateBoundary(r.trainer); err != nil {
				t.Fatal("writer failure damaged completed boundary", err)
			}
			if _, err := os.Stat(filepath.Join(dir, "current.json")); !os.IsNotExist(err) {
				t.Fatal("failed checkpoint selected a generation", err)
			}
			if files, _ := filepath.Glob(filepath.Join(dir, ".runner-serialise-*.tmp")); len(files) != 0 {
				t.Fatal("scratch file retained")
			}
			if kind == "existing-generation" {
				contents, err := os.ReadFile(filepath.Join(dir, "rejected.model.bin"))
				if err != nil || string(contents) != "preserved generation" {
					t.Fatal("existing generation overwritten", err)
				}
			}
		})
	}
}
