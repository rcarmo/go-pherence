package training

import "testing"

func TestTrainingEmbeddingAdmission(t *testing.T) {
	for mode := uint32(0); mode < 4; mode++ {
		a, n, err := EmbeddingShape(3, 2, 4, mode)
		if err != nil || (mode == 0 && (a != 8 || n != 6)) || ((mode == 1 || mode == 2) && (a != 6 || n != 8)) || (mode == 3 && (a != 6 || n != 6)) {
			t.Fatal(a, n, err)
		}
	}
	for _, s := range [][4]int{{0, 2, 4, 0}, {1025, 2, 4, 0}, {1, 0, 4, 0}, {1, 16385, 4, 0}, {1, 2, 0, 0}, {1, 2, 65537, 0}, {1, 16384, 65536, 0}, {1, 2, 4, 4}} {
		if _, _, err := EmbeddingShape(s[0], s[1], s[2], uint32(s[3])); err == nil {
			t.Fatal(s)
		}
	}
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{2}, Values: []float32{0, 2}}, {Shape: []int{4}}}, Commands: []Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 2, Width: 2, Parts: 3}}, Outputs: []int{2}}
	}
	if err := makeTape().Validate(100); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Tensors[1].Values = nil }, func(t *Tape) { t.Tensors[1].Values[0] = -.5 }, func(t *Tape) { t.Tensors[1].Values[0] = 3 }, func(t *Tape) { t.Tensors[1].Values[0] = 1.5 }, func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Commands[0].Mode = 2 }, func(t *Tape) { t.Tensors[2].Shape = []int{2, 2} }} {
		tape := makeTape()
		mutate(&tape)
		if tape.Validate(100) == nil {
			t.Fatal("bad tape")
		}
	}
}
