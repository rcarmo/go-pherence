package training

import (
	"context"
	"reflect"
	"testing"
)

func storageTestTape() Tape {
	return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}, {Shape: []int{1, 2}}, {Shape: []int{2}}, {Shape: []int{2}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "reshape", Inputs: []int{2}, Output: 3}, {Op: "vector", Inputs: []int{2, 0}, Output: 4, Scale: 1}, {Op: "vector", Inputs: []int{4, 1}, Output: 5, Scale: 1}, {Op: "vector", Inputs: []int{5, 1}, Output: 6, Scale: 1}}, Outputs: []int{6}}
}
func TestTrainingTapeStorageLifetimes(t *testing.T) {
	tape := storageTestTape()
	p, err := PlanTapeStorage(tape, 56)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots[2] != p.Slots[3] || p.Slots[2] != p.Slots[5] || p.Slots[4] != p.Slots[6] || len(p.Elements) != 4 || p.Bytes != 32 {
		t.Fatal("slots", p)
	}
	// Every command input/output must stay disjoint, despite reshape families.
	for _, c := range tape.Commands {
		if c.Op == "reshape" {
			continue
		}
		for _, id := range c.Inputs {
			if p.Slots[c.Output] == p.Slots[id] {
				t.Fatal("dispatch alias", c)
			}
		}
	}
	// Output alias pins canonical root beyond its final command consumer.
	tape.Outputs = append(tape.Outputs, 3)
	p, err = PlanTapeStorage(tape, 56)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots[2] == p.Slots[5] || p.Slots[2] == p.Slots[6] {
		t.Fatal("declared alias output overwritten", p)
	}
	// Longer reshape chains propagate all consumers to the root.
	tape = storageTestTape()
	tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{2}})
	tape.Commands = append(tape.Commands, Command{Op: "reshape", Inputs: []int{3}, Output: 7})
	tape.Outputs = []int{6, 7}
	p, err = PlanTapeStorage(tape, 64)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots[7] != p.Slots[2] || p.Slots[5] == p.Slots[2] {
		t.Fatal("reshape chain lifetime")
	}
	if _, err = PlanTapeStorage(tape, 63); err == nil {
		t.Fatal("full logical ceiling bypass")
	}
}
func TestTrainingTapeStorageConstantsAdmission(t *testing.T) {
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{2}}, {Shape: []int{1}, Values: []float32{3}}, {Shape: []int{1}}}, Outputs: []int{0}}
	p, err := PlanTapeStorage(tape, 12)
	if err != nil || !reflect.DeepEqual(p.Slots, []int{0, 1, 2}) || p.Bytes != 12 {
		t.Fatal("constant/dead storage", p, err)
	}
	bad := storageTestTape()
	bad.Commands[3].Inputs[0] = 6
	if _, err = PlanTapeStorage(bad, 1<<20); err == nil {
		t.Fatal("dependency")
	}
	bad = storageTestTape()
	bad.Tensors[0].Values = nil
	if _, err = PlanTapeStorage(bad, 1<<20); err == nil {
		t.Fatal("uninitialised")
	}
	if _, err = PlanTapeStorage(Tape{}, 1); err == nil {
		t.Fatal("empty")
	}
	// Different extents never share a slot; future output retention matters even
	// if a same-sized earlier interval otherwise permits reuse.
	tape = storageTestTape()
	tape.Tensors[3].Shape = []int{2}
	tape.Outputs = []int{2, 4, 5, 6}
	p, err = PlanTapeStorage(tape, 56)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Elements) != 6 {
		t.Fatal("retained independent outputs", p)
	}
}

func TestTrainingTapeStorageDifferentExtents(t *testing.T) {
	tpe := Tape{Tensors: []TensorSpec{{Shape: []int{3}, Values: []float32{1, 2, 3}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{2}}, {Shape: []int{3}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Mode: 1}, {Op: "vector", Inputs: []int{0, 1}, Output: 3, Mode: 1}}, Outputs: []int{3}}
	p, err := PlanTapeStorage(tpe, 36)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots[2] == p.Slots[3] {
		t.Fatal("different extents share storage")
	}
}

func TestTrainingTapeStorageReuseCapability(t *testing.T) {
	state, driver := requireResident(t)
	defer state.Close()
	tape := storageTestTape()
	before := driver.events["upload"]
	if _, err := state.RunReusePrunedUnboundTape(context.Background(), tape, tape.Outputs, 1<<20); err == nil || driver.events["upload"] != before {
		t.Fatal("missing reuse driver")
	}
}
