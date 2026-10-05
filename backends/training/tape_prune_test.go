package training

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func pruneFixture() Tape {
	return Tape{Tensors: []TensorSpec{
		{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}},
		{Shape: []int{2}, Values: []float32{7, 8}}, {Shape: []int{2}}, {Shape: []int{1, 2}}, {Shape: []int{2}},
	}, Commands: []Command{
		{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .5},
		{Op: "vector", Inputs: []int{2, 3}, Output: 4, Scale: 2},
		{Op: "reshape", Inputs: []int{2}, Output: 5},
		{Op: "vector", Inputs: []int{0, 3}, Output: 6, Scale: -1},
	}, Outputs: []int{2, 4, 5, 6}}
}
func TestTrainingTapePruneDependencies(t *testing.T) {
	source := pruneFixture()
	compact, mapping, err := PruneTape(source, []int{5, 2, 5}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(compact.Commands) != 2 || len(compact.Tensors) != 4 || !reflect.DeepEqual(mapping, []int{0, 1, 2, -1, -1, 3, -1}) || !reflect.DeepEqual(compact.Outputs, []int{3, 2, 3}) {
		t.Fatal("dependency remap", compact, mapping)
	}
	if compact.Commands[0].Scale != .5 || compact.Commands[1].Op != "reshape" || !reflect.DeepEqual(compact.Commands[1].Inputs, []int{2}) {
		t.Fatal("command semantics")
	}
	compact.Commands[0].Inputs[0] = 123
	compact.Tensors[0].Shape[0] = 99
	compact.Outputs[0] = 321
	if source.Commands[0].Inputs[0] != 0 || source.Tensors[0].Shape[0] != 2 || source.Outputs[0] != 2 {
		t.Fatal("metadata borrowed")
	}
	// Multiple roots retain shared fanout once, and original command order.
	compact, _, err = PruneTape(source, []int{4, 6}, 1024)
	if err != nil || len(compact.Commands) != 3 || compact.Commands[0].Scale != .5 || compact.Commands[1].Scale != 2 || compact.Commands[2].Scale != -1 {
		t.Fatal("multi-root closure", err)
	}
	// Constant-only roots retain exactly the requested constant, no dispatch.
	source.Outputs = append(source.Outputs, 0)
	compact, mapping, err = PruneTape(source, []int{0}, 1024)
	if err != nil || len(compact.Commands) != 0 || len(compact.Tensors) != 1 || mapping[3] != -1 || mapping[0] != 0 {
		t.Fatal("constant fallback", err)
	}
}
func TestTrainingTapePruneAdmission(t *testing.T) {
	for _, roots := range [][]int{nil, {}, {-1}, {7}, {0}, {3}} {
		if _, _, err := PruneTape(pruneFixture(), roots, 1024); err == nil {
			t.Fatal("bad root", roots)
		}
	}
	if _, _, err := PruneTape(pruneFixture(), []int{2}, 1); err == nil {
		t.Fatal("source budget not checked")
	}
	tape := pruneFixture()
	tape.Commands[3].Op = "unknown"
	if _, _, err := PruneTape(tape, []int{2}, 1024); err == nil {
		t.Fatal("malformed dead command removed")
	}
	tape = pruneFixture()
	tape.Commands[3].Inputs[0] = 4
	tape.Commands[0].Inputs[0] = 4
	if _, _, err := PruneTape(tape, []int{6}, 1024); err == nil {
		t.Fatal("invalid ordering removed")
	}
}
func TestTrainingResidentPrunedTapeInventory(t *testing.T) {
	s, d := requireBoundResident(t)
	tape, bindings := residentBindingFixture()
	// A dead independent parameter and its gradient are still inventory-admitted.
	tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{99}}, TensorSpec{Shape: []int{1}, Values: []float32{1}}, TensorSpec{Shape: []int{1}})
	tape.Commands = append(tape.Commands, Command{Op: "vector", Inputs: []int{3, 4}, Output: 5, Scale: 1})
	tape.Outputs = append(tape.Outputs, 5)
	bindings["b"] = ResidentTapeBinding{Parameter: 3, Gradient: 5}
	values, err := s.RunPrunedTape(context.Background(), tape, bindings, []int{2}, false, 1, 1024)
	if err != nil || values[0][0] != 1.3 || len(d.retained.tape.Commands) != 1 || len(d.retained.tape.Tensors) != 3 {
		t.Fatal(values, err)
	}
	if s.dirty || s.entries[0].ready || s.entries[1].ready {
		t.Fatal("forward accumulated")
	}
	// Accumulation retains ALL parameter gradients even without downloads.
	if _, err := s.RunPrunedTape(context.Background(), tape, bindings, nil, true, 1, 1024); err != nil {
		t.Fatal(err)
	}
	if !s.entries[0].ready || !s.entries[1].ready || len(d.retained.tape.Commands) != 2 {
		t.Fatal("gradient root lost")
	}
	bad := map[string]ResidentTapeBinding{"a": bindings["a"], "unknown": bindings["b"]}
	other, _ := requireBoundResident(t)
	if _, err := other.RunPrunedTape(context.Background(), tape, bad, []int{2}, false, 1, 1024); err == nil {
		t.Fatal("dead unknown parameter hidden")
	}
}
func TestTrainingResidentPrunedUnboundRetention(t *testing.T) {
	for _, failure := range []string{"construction", "nil", "run", "download", "count", "extent", "nonfinite", "close"} {
		t.Run(failure, func(t *testing.T) {
			s, d := requireBoundResident(t)
			d.tapeFailure = failure
			tape, _ := residentBindingFixture()
			if _, err := s.RunPrunedUnboundTape(context.Background(), tape, []int{2}, 1024); err == nil {
				t.Fatal("failure admitted")
			}
			if s.borrowed || s.terminal == nil || s.Close() == nil {
				t.Fatal("failure owner lost")
			}
		})
	}
	s, d := requireBoundResident(t)
	d.hook = true
	tape, _ := residentBindingFixture()
	if _, err := s.RunPrunedUnboundTape(context.Background(), tape, []int{2}, 1024); err != nil {
		t.Fatal(err)
	}
	if s.dirty || !d.retained.closed {
		t.Fatal("utility tape mutated state")
	}
}
func TestTrainingTapePruneConstantAdmission(t *testing.T) {
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}}, Outputs: []int{0}}
	if err := tape.Validate(8); err != nil {
		t.Fatal(err)
	}
	if _, _, err := PruneTape(tape, []int{0}, 8); err != nil {
		t.Fatal(err)
	}
	tape.Tensors[0].Values = nil
	if tape.Validate(8) == nil {
		t.Fatal("constant-only uninitialized output admitted")
	}
	s, d := requireBoundResident(t)
	source := pruneFixture()
	source.Outputs = append(source.Outputs, 0)
	if _, err := s.RunPrunedUnboundTape(context.Background(), source, []int{0}, 1024); err != nil {
		t.Fatal(err)
	}
	if len(d.retained.tape.Commands) != 0 || len(d.retained.tape.Tensors) != 1 {
		t.Fatal("constant-only pruning ran dead commands")
	}
}
func TestTrainingResidentPrunedIndependentGradient(t *testing.T) {
	s, d := requireBoundResident(t)
	// Gradient of a linear bias may be independent of its parameter. The live
	// gradient must accumulate even when the parameter constant is dead.
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{1, 2}, Output: 3, Scale: 1}}, Outputs: []int{3}}
	bindings := map[string]ResidentTapeBinding{"a": {Parameter: 0, Gradient: 3}}
	if _, err := s.RunPrunedTape(context.Background(), tape, bindings, nil, true, .5, 1024); err != nil {
		t.Fatal(err)
	}
	if len(d.retained.tape.Tensors) != 3 || len(d.retained.tape.Commands) != 1 || !s.entries[0].ready {
		t.Fatal("dead parameter/live gradient remap")
	}
	for i, want := range []float32{2, 3} {
		if s.entries[0].gradient.(*residentHostBuffer).values[i] != want {
			t.Fatal("gradient accumulation")
		}
	}
}
func TestTrainingResidentPrunedBoundRetention(t *testing.T) {
	for _, category := range []string{"construction", "nil", "run", "output", "download", "count", "extent", "nonfinite", "close", "admission", "accumulate"} {
		t.Run(category, func(t *testing.T) {
			s, d := requireBoundResident(t)
			before, err := s.Snapshot(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			d.tapeFailure = category
			if category == "accumulate" {
				d.fail, d.at, d.sentinel = "accumulate", 1, errors.New("accumulate")
			}
			tape, bindings := residentBindingFixture()
			if _, err := s.RunPrunedTape(context.Background(), tape, bindings, []int{2}, true, 1, 1024); err == nil {
				t.Fatal("failure admitted")
			}
			if s.borrowed {
				t.Fatal("borrow stuck")
			}
			if category == "admission" {
				after, err := s.Snapshot(context.Background())
				if err != nil || !reflect.DeepEqual(before, after) || !d.retained.closed {
					t.Fatal("pure preflight mutated")
				}
				return
			}
			if s.terminal == nil || s.Close() == nil {
				t.Fatal("uncertain owner lost")
			}
			if category != "nil" && len(s.retained) != 1 {
				t.Fatal("tape not retained")
			}
		})
	}
}
