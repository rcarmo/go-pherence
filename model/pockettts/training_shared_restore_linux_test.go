//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func BenchmarkPocketSharedFreshRestoreStorage(b *testing.B) {
	raw, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		b.Fatal(err)
	}
	var oracle trainingStepOracle
	if err := json.Unmarshal(raw, &oracle); err != nil {
		b.Fatal(err)
	}
	for _, mode := range []string{"public-restore-plus-state", "owned-shared-restore"} {
		b.Run(mode, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				lm, flow, w := trainingStepModelsFromOracle(b, oracle)
				w.Layers = []LinearF32{{In: 2, Out: 256, Weight: make([]float32, 512), Bias: make([]float32, 256)}, {In: 256, Out: 256, Weight: make([]float32, 65536), Bias: make([]float32, 256)}, {In: 256, Out: 1, Weight: make([]float32, 256), Bias: make([]float32, 1)}}
				tr, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
				if err != nil {
					b.Fatal(err)
				}
				source, err := tr.State()
				if err != nil {
					b.Fatal(err)
				}
				source.Step = 17
				for _, p := range source.Params {
					source.Adam = append(source.Adam, NamedAdamState{Name: p.Name, M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))})
				}
				backend := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
				b.StartTimer()
				if mode == "owned-shared-restore" {
					err = backend.restoreSharedOwnedState(tr, &source)
				} else {
					err = tr.LoadState(source)
					if err == nil {
						backend.host, err = tr.State()
					}
				}
				if err != nil {
					b.Fatal(err)
				}
				if tr.StepCount != 17 || len(backend.host.Adam) != len(backend.host.Params) {
					b.Fatal("restore storage")
				}
			}
		})
	}
}

func decodedSharedRestoreFixture(t *testing.T) FullTrainingState {
	t.Helper()
	r := requireRunner(t)
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	state := mustFullState(t, r.trainer)
	state.Step = 17 // restore admits arbitrary positive completed boundary
	var data bytes.Buffer
	if err := writeFullTrainingBinary(&data, state); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadFullTrainingStateBinary(bytes.NewReader(data.Bytes()), int64(data.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return decoded
}

func TestPocketSharedOwnedRestoreParityAndConsumption(t *testing.T) {
	state := decodedSharedRestoreFixture(t)
	refs := state
	live, reference := requireRunner(t), requireRunner(t)
	beforeEMA := map[string][]float32{}
	for name, x := range live.trainer.EMA {
		beforeEMA[name] = x
	}
	if err := reference.trainer.LoadState(state); err != nil {
		t.Fatal(err)
	}
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	if err := b.restoreSharedOwnedState(live.trainer, &state); err != nil {
		t.Fatal(err)
	}
	if state.Params != nil || state.Step != 0 || b.state != nil || !sameResidentLiveBoundary(b.host, live.trainer) || !reflect.DeepEqual(mustFullState(t, reference.trainer), mustFullState(t, live.trainer)) {
		t.Fatal("shared owned restore parity/source consumption")
	}
	for i, p := range b.host.Params {
		if &p.Values[0] != &refs.Params[i].Values[0] || f32SlicesOverlap(p.Values, live.trainer.params[p.Name]) {
			t.Fatal("decoded parameter ownership")
		}
	}
	for i, m := range b.host.Adam {
		if &m.M[0] != &refs.Adam[i].M[0] || &m.V[0] != &refs.Adam[i].V[0] || f32SlicesOverlap(m.M, live.trainer.Moments[m.Name].M) || f32SlicesOverlap(m.V, live.trainer.Moments[m.Name].V) {
			t.Fatal("decoded moment ownership")
		}
	}
	for i, p := range b.host.EMA {
		if &p.Values[0] != &refs.EMA[i].Values[0] || &live.trainer.EMA[p.Name][0] != &beforeEMA[p.Name][0] {
			t.Fatal("EMA destination reused or source ownership")
		}
	}
	if err := b.ValidateBoundary(live.trainer); err != nil {
		t.Fatal(err)
	}
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	if err := b.PrepareTrainingStep(context.Background(), live.trainer); err != nil {
		t.Fatal(err)
	}
	if b.sharedInitialMoments != nil {
		t.Fatal("restored moments duplicated")
	}
	for i, p := range b.host.Params {
		if &p.Values[0] != &refs.Params[i].Values[0] {
			t.Fatal("prepare took extra host state")
		}
	}
	// Only synchronous initial uploads occurred, no graph submission.
	b.active = false
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPocketSharedOwnedRestoreRejectionAtomic(t *testing.T) {
	for _, kind := range []string{"version", "step-zero", "optimizer", "EMA-decay", "negativeV", "missingAdam", "alias-host", "alias-live", "buffer", "fresh-only", "owner", "nil"} {
		t.Run(kind, func(t *testing.T) {
			state := decodedSharedRestoreFixture(t)
			live := requireRunner(t)
			b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
			source := &state
			switch kind {
			case "version":
				state.Version = 0
			case "step-zero":
				state.Step = 0
			case "optimizer":
				state.AdamW.Epsilon *= 2
			case "EMA-decay":
				state.EMADecay = .5
			case "negativeV":
				state.Adam[0].V[0] = -1
			case "missingAdam":
				state.Adam = nil
			case "alias-host":
				state.Adam[0].M = state.Params[0].Values
			case "alias-live":
				state.Params[0].Values = live.trainer.params[state.Params[0].Name]
			case "buffer":
				state.Buffers[0].Name = "foreign"
			case "fresh-only":
				for _, p := range state.Params {
					live.trainer.Moments[p.Name] = NamedAdamState{Name: p.Name, M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))}
				}
			case "owner":
				b.terminal = errors.New("fixture terminal")
			case "nil":
				source = nil
			}
			before := mustFullState(t, live.trainer)
			raw := state
			if err := b.restoreSharedOwnedState(live.trainer, source); err == nil {
				t.Fatal("bad restore accepted")
			}
			if b.host.Params != nil || b.state != nil || !reflect.DeepEqual(before, mustFullState(t, live.trainer)) || !reflect.DeepEqual(raw, state) {
				t.Fatal("failed restore consumed source/mutated live")
			}
		})
	}
	if (*ResidentTrainingBackend)(nil).restoreSharedOwnedState(nil, nil) == nil {
		t.Fatal("nil backend")
	}
}

func TestPocketSharedRunnerOwnedRestoreAndPrepare(t *testing.T) {
	original := requireRunner(t)
	if _, err := original.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	_, _ = installResidentGraph(t, original.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	original.run.core.identity.Backend = "vulkan-resident-sharedworkspace-hybrid-v1"
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := original.Checkpoint(dir, "owned", limits); err != nil {
		t.Fatal(err)
	}
	fresh := requireRunner(t)
	b, _ := installResidentGraph(t, fresh.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	fresh.run.core.identity.Backend = b.Scope
	if err := fresh.Restore(dir, limits); err != nil {
		t.Fatal(err)
	}
	if b.state != nil || !sameResidentLiveBoundary(b.host, fresh.trainer) || !reflect.DeepEqual(mustFullState(t, original.trainer), mustFullState(t, fresh.trainer)) {
		t.Fatal("runner owned restore")
	}
	if !bytes.Equal(snapshotRunFixture(t, original.run), snapshotRunFixture(t, fresh.run)) {
		t.Fatal("run restore association")
	}
	if _, err := fresh.Checkpoint(dir, "before-native", limits); err != nil {
		t.Fatal("restored owned checkpoint before native prepare", err)
	}
	refs := b.host
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	if err := b.PrepareTrainingStep(context.Background(), fresh.trainer); err != nil {
		t.Fatal(err)
	}
	for i, p := range b.host.Params {
		if &p.Values[0] != &refs.Params[i].Values[0] {
			t.Fatal("restored Prepare cloned host")
		}
	}
	b.active = false
	if _, err := fresh.Checkpoint(dir, "restored", limits); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPocketSharedRestoredBoundaryMutationRejected(t *testing.T) {
	live := requireRunner(t)
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	state := decodedSharedRestoreFixture(t)
	if err := b.restoreSharedOwnedState(live.trainer, &state); err != nil {
		t.Fatal(err)
	}
	name := live.trainer.names[0]
	live.trainer.params[name][0] += .1
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	if b.ValidateBoundary(live.trainer) == nil || b.PrepareTrainingStep(context.Background(), live.trainer) == nil || d.allocations != 0 {
		t.Fatal("restored mutation crossed native allocation")
	}
	if _, err := b.sharedCheckpointState(live.trainer); err == nil {
		t.Fatal("mutated pending restored checkpoint borrowed")
	}
}

func TestPocketSharedRestoredNextSnapshotUsesOwnedBoundary(t *testing.T) {
	live := requireRunner(t)
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	state := decodedSharedRestoreFixture(t)
	if err := b.restoreSharedOwnedState(live.trainer, &state); err != nil {
		t.Fatal(err)
	}
	refs := b.host
	// Independent next completed snapshot source, no native graph execution.
	reference := requireRunner(t)
	if err := reference.trainer.LoadState(b.host); err != nil {
		t.Fatal(err)
	}
	batch, samples := tinyFlowLMBatch(), tinyTrainingStepSamples()
	_, g, err := PocketTrainingStep(reference.trainer.FlowLM, reference.trainer.Flow, reference.trainer.Weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := reference.clipGradients(g); err != nil {
		t.Fatal(err)
	}
	if err := reference.trainer.Step(g); err != nil {
		t.Fatal(err)
	}
	desired := mustFullState(t, reference.trainer)
	snapshot, err := sharedInitialSnapshot(desired)
	if err != nil {
		t.Fatal(err)
	}
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	b.state, err = train.NewSharedWorkspaceResidentState(context.Background(), d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.loadSharedCompletedSnapshot(context.Background(), live.trainer); err != nil {
		t.Fatal(err)
	}
	if live.trainer.StepCount != 18 || !reflect.DeepEqual(desired, mustFullState(t, live.trainer)) || !sameResidentLiveBoundary(b.host, live.trainer) {
		t.Fatal("restored next completed load parity")
	}
	for i, m := range refs.Adam {
		if &m.M[0] != &b.host.Adam[i].M[0] || &m.V[0] != &b.host.Adam[i].V[0] {
			t.Fatal("restored host moments replaced")
		}
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPocketSharedOwnedRestoreNoEMAAndInvalidLiveTopology(t *testing.T) {
	state := decodedSharedRestoreFixture(t)
	state.EMADecay = 0
	state.EMA = nil
	live := requireRunner(t)
	live.trainer.EMADecay = 0
	live.trainer.EMA = nil
	b, _ := installResidentGraph(t, live.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	if err := b.restoreSharedOwnedState(live.trainer, &state); err != nil {
		t.Fatal(err)
	}
	if live.trainer.EMA != nil || len(b.host.EMA) != 0 || !sameResidentLiveBoundary(b.host, live.trainer) {
		t.Fatal("disabled EMA restore contract")
	}
	state = decodedSharedRestoreFixture(t)
	invalid := requireRunner(t)
	ib, _ := installResidentGraph(t, invalid.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	invalid.trainer.Weighting.Layers = append(invalid.trainer.Weighting.Layers, invalid.trainer.Weighting.Layers[0])
	if err := ib.restoreSharedOwnedState(invalid.trainer, &state); err == nil || state.Params == nil || invalid.trainer.StepCount != 0 || ib.host.Params != nil {
		t.Fatal("bad live topology restore mutated source/state")
	}
}

func TestPocketSharedRunnerStepZeroRestoreRemainsPublic(t *testing.T) {
	create := func() (*TrainingRunner, *ResidentTrainingBackend) {
		r := requireRunner(t)
		b, _ := installResidentGraph(t, r.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
		r.run.core.identity.Backend = b.Scope
		return r, b
	}
	source, _ := create()
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := source.Checkpoint(dir, "initial", limits); err != nil {
		t.Fatal(err)
	}
	fresh, b := create()
	if err := fresh.Restore(dir, limits); err != nil {
		t.Fatal(err)
	}
	if b.host.Params != nil || b.trainer != nil || b.state != nil || fresh.trainer.StepCount != 0 || !reflect.DeepEqual(mustFullState(t, source.trainer), mustFullState(t, fresh.trainer)) {
		t.Fatal("step0 shared restore changed public path")
	}
}

var _ train.ResidentDriver = (*sharedSnapshotOnlyDriver)(nil)
