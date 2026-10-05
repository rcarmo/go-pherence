package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestPTXTrainingOfflineHandoffPreflight(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	d := &PTXResidentDriver{Operator: &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_vector": 21}}}}
	a, b := &ptxResidentBuffer{buffer: &Buffer{Ptr: 4096, Size: 8}}, &ptxResidentBuffer{buffer: &Buffer{Ptr: 8192, Size: 8}}
	if err := d.ValidateHandoffTape(context.Background(), tape, map[int]train.ResidentBuffer{0: a, 1: b}, 100); err != nil {
		t.Fatal(err)
	}
	external, err := train.WithExternalTapeInputs(tape, []int{0}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.ValidateHandoffTape(context.Background(), external, map[int]train.ResidentBuffer{0: a, 1: b}, 100); err != nil {
		t.Fatal(err)
	}
	if d.ValidateHandoffTape(context.Background(), external, nil, 100) == nil {
		t.Fatal("missing external preflight")
	}
	if owner, err := NewTrainingTape(d.Operator, external, 100); err == nil || owner != nil {
		t.Fatal("unbound external allocated native storage")
	}
	dead := external
	dead.Tensors = append(append([]train.TensorSpec(nil), external.Tensors...), train.TensorSpec{Shape: []int{1}, External: true})
	if d.ValidateHandoffTape(context.Background(), dead, map[int]train.ResidentBuffer{0: a}, 100) == nil {
		t.Fatal("dead external omitted")
	}
	for _, bindings := range []map[int]train.ResidentBuffer{{-1: a}, {2: a}, {0: nil}, {0: &ptxResidentBuffer{buffer: &Buffer{Ptr: 4096, Size: 4}}}, {0: a, 1: a}, {0: &ptxResidentBuffer{buffer: &Buffer{Ptr: 4096, Size: 8}, sourceTape: &TrainingTape{closed: true}}}} {
		if d.ValidateHandoffTape(context.Background(), tape, bindings, 100) == nil {
			t.Fatal("bad bindings")
		}
	}
	if (*PTXResidentDriver)(nil).ValidateHandoffTape(context.Background(), tape, nil, 100) == nil {
		t.Fatal("nil")
	}
	if d.ValidateHandoffTape(nil, tape, nil, 100) == nil {
		t.Fatal("context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.ValidateHandoffTape(ctx, tape, nil, 100) == nil {
		t.Fatal("cancel")
	}
	invalid := tape
	invalid.Tensors = append(append([]train.TensorSpec(nil), tape.Tensors...), train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if d.ValidateHandoffTape(context.Background(), invalid, nil, 100) == nil {
		t.Fatal("full source")
	}
	closed := &PTXResidentDriver{Operator: &TrainingF32{}}
	if closed.ValidateHandoffTape(context.Background(), tape, nil, 100) == nil {
		t.Fatal("missing kernel")
	}
}

func TestPTXTrainingOfflineHandoffExternalOwner(t *testing.T) {
	source := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, External: true}}, Outputs: []int{0}}
	borrowed := &Buffer{Ptr: 4096, Size: 8}
	owner, err := newTrainingTapeBound(&TrainingF32{}, source, map[int]*Buffer{0: borrowed}, 8, true, true)
	if err != nil || owner == nil || len(owner.owned) != 0 || owner.buffers[0] != borrowed {
		t.Fatal("borrowed-only construction", err)
	}
	if err := owner.runDevice(); err != nil {
		t.Fatal(err)
	}
	if owner.Close() != nil || borrowed.Ptr != 4096 {
		t.Fatal("external source freed")
	}
}
