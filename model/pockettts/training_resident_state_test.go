package pockettts

import (
	"math"
	"reflect"
	"testing"
)

func TestTrainingResidentStateCheckpointConversion(t *testing.T) {
	for _, ema := range []float32{0, .9} {
		lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
		flow.Condition = tinyLinear(4, 4, -.02)
		trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), ema)
		if err != nil {
			t.Fatal(err)
		}
		initial, err := trainer.State()
		if err != nil {
			t.Fatal(err)
		}
		snapshot, err := ResidentSnapshotFromFullState(initial)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := FullStateFromResidentSnapshot(initial, snapshot)
		if err != nil {
			t.Fatal(err)
		}
		if len(decoded.Adam) != len(initial.Params) || len(decoded.Buffers) != len(initial.Buffers) || decoded.Step != 0 {
			t.Fatal("initial state conversion")
		}
		for _, p := range decoded.Params {
			for _, want := range initial.Params {
				if p.Name == want.Name {
					assertSliceClose(t, "resident initial "+p.Name, p.Values, want.Values, 1e-7)
				}
			}
		}
		if err := trainer.LoadState(decoded); err != nil {
			t.Fatal(err)
		}
		decoded.Params[0].Values[0] = 42
		fresh, err := FullStateFromResidentSnapshot(initial, snapshot)
		if err != nil || fresh.Params[0].Values[0] == 42 {
			t.Fatal("conversion borrowed arrays")
		}
		snapshot.Step = 1
		state, err := FullStateFromResidentSnapshot(initial, snapshot)
		if err != nil {
			t.Fatal(err)
		}
		if state.Step != 1 || len(state.Adam) != len(state.Params) || (ema == 0 && len(state.EMA) != 0) {
			t.Fatal("completed snapshot metadata")
		}
		before := mustFullState(t, trainer)
		for _, mutate := range []func(){func() { snapshot.Config.LearningRate *= 2 }, func() { delete(snapshot.Parameters, initial.Params[0].Name) }} {
			original, err := ResidentSnapshotFromFullState(initial)
			if err != nil {
				t.Fatal(err)
			}
			snapshot = original
			mutate()
			if _, err := FullStateFromResidentSnapshot(initial, snapshot); err == nil {
				t.Fatal("invalid snapshot accepted")
			}
			if !reflect.DeepEqual(before, mustFullState(t, trainer)) {
				t.Fatal("rejected conversion mutated trainer")
			}
		}
	}
}
func TestTrainingResidentStateCheckpointRejection(t *testing.T) {
	if _, err := ResidentSnapshotFromFullState(FullTrainingState{}); err == nil {
		t.Fatal("invalid full state")
	}
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	state := mustFullState(t, trainer)
	snapshot, err := ResidentSnapshotFromFullState(state)
	if err != nil {
		t.Fatal(err)
	}
	p := snapshot.Parameters[state.Params[0].Name]
	p.V[0] = -1
	if _, err := FullStateFromResidentSnapshot(state, snapshot); err == nil {
		t.Fatal("negative second moment")
	}
	snapshot, err = ResidentSnapshotFromFullState(state)
	if err != nil {
		t.Fatal(err)
	}
	p = snapshot.Parameters[state.Params[0].Name]
	p.Parameter[0] = float32(math.NaN())
	if _, err := FullStateFromResidentSnapshot(state, snapshot); err == nil {
		t.Fatal("nonfinite parameter")
	}
}

func TestTrainingResidentStateInventory(t *testing.T) {
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	state := mustFullState(t, trainer)
	snapshot, err := ResidentSnapshotFromFullState(state)
	if err != nil {
		t.Fatal(err)
	}
	key := state.Params[0].Name
	p := snapshot.Parameters[key]
	delete(snapshot.Parameters, key)
	snapshot.Parameters["unknown"] = p
	if _, err := FullStateFromResidentSnapshot(state, snapshot); err == nil {
		t.Fatal("same count wrong names")
	}
	if _, err := FullStateFromResidentSnapshot(FullTrainingState{}, snapshot); err == nil {
		t.Fatal("invalid template")
	}
	state.Adam = []NamedAdamState{{Name: key, M: make([]float32, len(state.Params[0].Values)), V: make([]float32, len(state.Params[0].Values))}}
	state.Adam[0].V[0] = -1
	if _, err := ResidentSnapshotFromFullState(state); err == nil {
		t.Fatal("negative imported moment")
	}
}
