package training

import (
	"context"
	"math"
	"testing"
)

func TestTrainingTapeZeroStorage(t *testing.T) {
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{float32(math.Copysign(0, -1)), 0}}, {Shape: []int{2}}, {Shape: []int{1, 2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 4, Scale: 1}, {Op: "reshape", Inputs: []int{1}, Output: 5}}, Outputs: []int{4, 5}}
	p, err := PlanZeroTapeStorage(tape, 48, nil)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots[0] != p.Slots[1] || p.Slots[5] != p.Slots[1] || p.Slots[0] == p.Slots[4] || p.Slots[3] == p.Slots[0] || p.Bytes != 32 {
		t.Fatal("zero storage contract", p)
	}
	old, err := PlanTapeStorage(tape, 48)
	if err != nil || old.Bytes != 40 || old.Slots[0] == old.Slots[1] {
		t.Fatal("legacy constants changed", old, err)
	}
	pinned, err := PlanZeroTapeStorage(tape, 48, map[int]bool{0: true})
	if err != nil || pinned.Slots[0] == pinned.Slots[1] || pinned.Bytes != 40 {
		t.Fatal("bound zero parameter shared", pinned, err)
	}
	if _, err = PlanZeroTapeStorage(tape, 47, nil); err == nil {
		t.Fatal("source ceiling bypass")
	}
	// Protected means allocation identity, not current numerical value. Several
	// zero-valued parameters remain disjoint even with a shared utility zero.
	all, err := PlanZeroTapeStorage(tape, 48, map[int]bool{0: true, 1: true})
	if err != nil || all.Slots[0] == all.Slots[1] {
		t.Fatal("protected identity")
	}
}
func TestTrainingTapeZeroAdmission(t *testing.T) {
	for _, x := range [][]float32{nil, {}, {float32(math.Copysign(0, -1))}, {1}, {float32(math.NaN())}} {
		if positiveZeroConstant(x) {
			t.Fatal("not positive-zero constant", x)
		}
	}
	if !positiveZeroConstant([]float32{0, 0}) {
		t.Fatal("positive zeros")
	}
	state, d := requireResident(t)
	defer state.Close()
	before := d.events["upload"]
	if _, err := state.RunZeroPrunedUnboundTape(context.Background(), storageTestTape(), []int{6}, 1<<20); err == nil || d.events["upload"] != before {
		t.Fatal("missing zero sharing capability")
	}
	// Constants-only output sharing does not eliminate declared output identity.
	p, err := PlanZeroTapeStorage(Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{0, 1}}, 8, nil)
	if err != nil || p.Bytes != 4 || p.Slots[0] != p.Slots[1] {
		t.Fatal("constant output sharing", p, err)
	}
}
