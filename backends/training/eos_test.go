package training

import (
	"context"
	"testing"
)

func TestTrainingEOSAdmission(t *testing.T) {
	for mode := uint32(0); mode < 3; mode++ {
		n, err := EOSShape(2, 1, mode, .1)
		want := 2
		if mode == 2 {
			want = 1
		}
		if err != nil || n != want {
			t.Fatal(n, err)
		}
	}
	for _, s := range [][3]int{{0, 1, 0}, {1025, 1, 0}, {1, 0, 0}, {1, 2, 0}, {1, 1, 3}} {
		if _, err := EOSShape(s[0], s[1], uint32(s[2]), 1); err == nil {
			t.Fatal(s)
		}
	}
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{0, -1}}, {Shape: []int{2}}}, Commands: []Command{{Op: "eos", Inputs: []int{0, 1}, Output: 2, Rows: 1, Scale: 1}}, Outputs: []int{2}}
	}
	if err := makeTape().Validate(100); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Tensors[1].Values = nil }, func(t *Tape) { t.Tensors[1].Values[0] = .5 }, func(t *Tape) { t.Commands[0].Rows = 2 }, func(t *Tape) { t.Commands[0].Mode = 3 }, func(t *Tape) { t.Commands[0].Scale = -1 }, func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Tensors[2].Shape = []int{2, 1} }} {
		tape := makeTape()
		mutate(&tape)
		if tape.Validate(100) == nil {
			t.Fatal("invalid")
		}
	}
}
func TestTrainingResidentUnboundTape(t *testing.T) {
	s, d := requireBoundResident(t)
	d.hook = true
	tape, _ := residentBindingFixture()
	values, err := s.RunUnboundTape(context.Background(), tape, []int{2}, 1024)
	if err != nil || values[0][0] != 100 || values[0][1] != 101 {
		t.Fatal("unbound values", values, err)
	}
	if s.dirty || s.entries[0].ready || !d.retained.closed || s.borrowed {
		t.Fatal("unbound tape changed resident state")
	}
	if _, err := s.RunTape(context.Background(), tape, nil, []int{2}, false, 1, 1024); err == nil {
		t.Fatal("bound API accepts empty bindings")
	}
	if _, err := s.RunUnboundTape(context.Background(), tape, []int{0}, 1024); err == nil {
		t.Fatal("undeclared output")
	}
}

func TestTrainingResidentUnboundFailureRetention(t *testing.T) {
	for _, category := range []string{"construction", "nil", "run", "download", "count", "extent", "nonfinite", "close"} {
		t.Run(category, func(t *testing.T) {
			s, d := requireBoundResident(t)
			d.tapeFailure = category
			tape, _ := residentBindingFixture()
			if _, err := s.RunUnboundTape(context.Background(), tape, []int{2}, 1024); err == nil {
				t.Fatal("accepted")
			}
			if s.terminal == nil || s.Close() == nil || s.borrowed {
				t.Fatal("lost owner")
			}
			if _, err := s.RunUnboundTape(context.Background(), tape, []int{2}, 1024); err == nil {
				t.Fatal("retry")
			}
		})
	}
}
