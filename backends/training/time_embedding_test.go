package training

import (
	"math"
	"testing"
)

func TestTrainingTimeEmbeddingAdmission(t *testing.T) {
	for mode := uint32(0); mode < 3; mode++ {
		n, err := TimeEmbeddingShape(4, mode)
		want := 8
		if mode == 2 {
			want = 1
		}
		if err != nil || n != want {
			t.Fatal(n, err)
		}
	}
	for _, s := range [][2]int{{0, 0}, {8193, 0}, {1, 3}} {
		if _, err := TimeEmbeddingShape(s[0], uint32(s[1])); err == nil {
			t.Fatal(s)
		}
	}
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{.2, 1}}, {Shape: []int{4}, Values: make([]float32, 4)}, {Shape: []int{4}, Values: make([]float32, 4)}, {Shape: []int{4}}}, Commands: []Command{{Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 4}}, Outputs: []int{4}}
	}
	if err := makeTape().Validate(100); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Tensors[0].Values = nil }, func(t *Tape) { t.Tensors[1].Values = nil }, func(t *Tape) { t.Tensors[0].Values[0] = float32(math.NaN()) }, func(t *Tape) { t.Tensors[0].Values[0] = float32(math.MaxFloat32) }, func(t *Tape) { t.Tensors[1].Values[0] = 8193 }, func(t *Tape) { t.Tensors[0].Values[0] = 1e19; t.Tensors[1].Values = []float32{0, 10} }, func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Commands[0].Mode = 3 }, func(t *Tape) { t.Tensors[4].Shape = []int{2, 2} }, func(t *Tape) { t.Tensors[2].Shape = []int{2, 2} }} {
		tape := makeTape()
		mutate(&tape)
		if tape.Validate(100) == nil {
			t.Fatal("bad tape")
		}
	}
}
