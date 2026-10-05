package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingProjectionTapeReference(t *testing.T) {
	for _, bias := range []bool{false, true} {
		l := LinearF32{In: 2, Out: 2, Weight: []float32{1, 2, -3, 4}}
		if bias {
			l.Bias = []float32{5, -1}
		}
		x, seed := []float32{2, -1, 3, 4}, []float32{1, 2, -1, 3}
		tape, err := BuildProjectionTrainingTape(l, "weights", "bias", x, seed, 2, 1024)
		if err != nil {
			t.Fatal(err)
		}
		if err := train.ValidateResidentTapeBindings(tape.Tape, tape.ResidentBindings, 1024); err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, tape.Tape)
		expected := []float32{0, -10, 11, 7}
		if bias {
			expected = []float32{5, -11, 16, 6}
		}
		// Independent integer matrix calculus: X=[[2,-1],[3,4]], seeds as above.
		assertSliceClose(t, "projection output", got[0], expected, 0)
		assertSliceClose(t, "projection input", got[1], []float32{-5, 10, -10, 10}, 0)
		assertSliceClose(t, "projection weight", got[2], []float32{-1, -5, 13, 10}, 0)
		if bias {
			assertSliceClose(t, "projection bias", got[3], []float32{0, 5}, 0)
		}
		x[0] = 42
		if tape.Tape.Tensors[0].Values[0] == 42 {
			t.Fatal("graph input borrowed")
		}
	}
}
func TestTrainingProjectionTapeAdmission(t *testing.T) {
	l := LinearF32{In: 2, Out: 1, Weight: []float32{1, 2}, Bias: []float32{.1}}
	x, seed := []float32{.2, .4}, []float32{.3}
	for _, mutate := range []func(*LinearF32){func(l *LinearF32) { l.In = 0 }, func(l *LinearF32) { l.Weight = nil }, func(l *LinearF32) { l.Weight[0] = float32(math.NaN()) }, func(l *LinearF32) { l.Bias = []float32{0, 1} }} {
		copy := l
		copy.Weight = append([]float32(nil), l.Weight...)
		mutate(&copy)
		if _, err := BuildProjectionTrainingTape(copy, "w", "b", x, seed, 1, 1024); err == nil {
			t.Fatal("bad linear")
		}
	}
	for _, names := range [][2]string{{"", "b"}, {"w", ""}, {"w", "w"}} {
		if _, err := BuildProjectionTrainingTape(l, names[0], names[1], x, seed, 1, 1024); err == nil {
			t.Fatal("names")
		}
	}
	if _, err := BuildProjectionTrainingTape(l, "w", "b", x, seed, 1, 1); err == nil {
		t.Fatal("budget")
	}
	if _, err := BuildProjectionTrainingTape(l, "w", "b", nil, seed, 1, 1024); err == nil {
		t.Fatal("input")
	}
	if _, err := BuildProjectionTrainingTape(l, "w", "b", x, []float32{float32(math.Inf(1))}, 1, 1024); err == nil {
		t.Fatal("nonfinite seed")
	}
	valid := ProjectionTrainingResult{Output: []float32{1}, DInput: []float32{2, 3}}
	if validateProjectionResult(valid, 1, 2, 1, x, seed) != nil {
		t.Fatal("valid boundary")
	}
	if validateProjectionResult(ProjectionTrainingResult{}, 1, 2, 1) == nil {
		t.Fatal("empty boundary")
	}
	if validateProjectionResult(ProjectionTrainingResult{Output: []float32{float32(math.NaN())}, DInput: []float32{0, 0}}, 1, 2, 1) == nil {
		t.Fatal("nonfinite output")
	}
	if validateProjectionResult(ProjectionTrainingResult{Output: x[:1], DInput: x}, 1, 2, 1) == nil {
		t.Fatal("overlap")
	}
	if validateProjectionResult(ProjectionTrainingResult{Output: seed, DInput: []float32{1, 2}}, 1, 2, 1, seed) == nil {
		t.Fatal("borrowed input")
	}
}
