package training

import "testing"

func TestTrainingLSDLossAdmission(t *testing.T) {
	for mode := uint32(0); mode < 2; mode++ {
		n, err := LSDLossShape(7, mode)
		want := 3
		if mode == 1 {
			want = 7
		}
		if err != nil || n != want {
			t.Fatal(n, err)
		}
	}
	for _, s := range [][2]int{{0, 0}, {16385, 0}, {1, 3}} {
		if _, err := LSDLossShape(s[0], uint32(s[1])); err == nil {
			t.Fatal(s)
		}
	}
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{0, 1}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{3}}}, Commands: []Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3}}, Outputs: []int{3}}
	}
	if err := makeTape().Validate(100); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Tensors[2].Values = nil }, func(t *Tape) { t.Tensors[2].Values[0] = 81 }, func(t *Tape) { t.Tensors[2].Values[0] = -81 }, func(t *Tape) { t.Commands[0].Mode = 2 }, func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Tensors[3].Shape = []int{3, 1} }} {
		tape := makeTape()
		mutate(&tape)
		if tape.Validate(100) == nil {
			t.Fatal("invalid")
		}
	}
}
