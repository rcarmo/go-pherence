package training

import "testing"

func TestTrainingReLUAdmission(t *testing.T) {
	for _, n := range []int{1, 257, 1 << 26} {
		for mode := uint32(0); mode < 2; mode++ {
			if err := ReLUShape(n, mode); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, n := range []int{0, -1, 1<<26 + 1} {
		if ReLUShape(n, 0) == nil {
			t.Fatal("invalid extent")
		}
	}
	if ReLUShape(1, 2) == nil {
		t.Fatal("invalid mode")
	}
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{3}, Values: []float32{-1, 0, 1}}, {Shape: []int{3}, Values: []float32{1, 2, 3}}, {Shape: []int{3}}}, Commands: []Command{{Op: "relu", Inputs: []int{0, 1}, Output: 2}}, Outputs: []int{2}}
	if err := tape.Validate(1024); err != nil {
		t.Fatal(err)
	}
	tape.Commands[0].Mode = 2
	if tape.Validate(1024) == nil {
		t.Fatal("invalid tape mode")
	}
	tape.Commands[0].Mode = 1
	tape.Commands[0].Inputs = nil
	if tape.Validate(1024) == nil {
		t.Fatal("invalid tape count")
	}
}
