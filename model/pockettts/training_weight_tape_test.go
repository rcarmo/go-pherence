package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingWeightTapeReference(t *testing.T) {
	// Independent scalar F64 affine/ReLU oracle and finite-difference VJP.
	m := &LSDWeightMLP{Layers: []LinearF32{{In: 2, Out: 3, Weight: []float32{1, -2, .5, 1, -1, .25}, Bias: []float32{.1, -.2, .3}}, {In: 3, Out: 1, Weight: []float32{.2, -.4, .7}, Bias: []float32{-.1}}}}
	input, seed := []float32{.2, .8, .7, .9, .4, .3}, []float32{.3, -.7, .4}
	tape, err := m.BuildWeightTrainingTape(input, seed, 3, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tape.Decode(interpretFlowTape(t, tape.Tape))
	if err != nil {
		t.Fatal(err)
	}
	if err := train.ValidateResidentTapeBindings(tape.Tape, tape.ResidentBindings, 1<<20); err != nil {
		t.Fatal(err)
	}
	reference := func(changed []float32, index int, delta float64) ([]float64, float64) {
		current := make([]float64, len(input))
		for i, x := range input {
			current[i] = float64(x)
			if len(changed) > 0 && &changed[0] == &input[0] && i == index {
				current[i] += delta
			}
		}
		for li, l := range m.Layers {
			out := make([]float64, 3*l.Out)
			for row := 0; row < 3; row++ {
				for o := 0; o < l.Out; o++ {
					bias := float64(l.Bias[o])
					if len(changed) > 0 && &changed[0] == &l.Bias[0] && o == index {
						bias += delta
					}
					sum := bias
					for k := 0; k < l.In; k++ {
						weight := float64(l.Weight[o*l.In+k])
						if len(changed) > 0 && &changed[0] == &l.Weight[0] && o*l.In+k == index {
							weight += delta
						}
						sum += current[row*l.In+k] * weight
					}
					if li < len(m.Layers)-1 && sum < 0 {
						sum = 0
					}
					out[row*l.Out+o] = sum
				}
			}
			current = out
		}
		var objective float64
		for i, x := range current {
			objective += x * float64(seed[i])
		}
		return current, objective
	}
	expected, _ := reference(nil, 0, 0)
	for i, x := range got.Output {
		if math.Abs(float64(x)-expected[i]) > 2e-7 {
			t.Fatal("weighting forward", i, x, expected[i])
		}
	}
	for i, x := range got.DInput {
		_, hi := reference(input, i, 1e-4)
		_, lo := reference(input, i, -1e-4)
		if math.Abs(float64(x)-(hi-lo)/2e-4) > 2e-6 {
			t.Fatal("weighting input VJP", i)
		}
	}
	names := []string{"flow.w_s_t.0.weight", "flow.w_s_t.0.bias", "flow.w_s_t.2.weight", "flow.w_s_t.2.bias"}
	values := [][]float32{m.Layers[0].Weight, m.Layers[0].Bias, m.Layers[1].Weight, m.Layers[1].Bias}
	for i, name := range names {
		for j, x := range got.Parameters[name] {
			_, hi := reference(values[i], j, 1e-4)
			_, lo := reference(values[i], j, -1e-4)
			if math.Abs(float64(x)-(hi-lo)/2e-4) > 2e-6 {
				t.Fatal("weighting parameter VJP", name, j)
			}
		}
	}
	input[0] = 42
	if tape.Tape.Tensors[0].Values[0] == 42 {
		t.Fatal("borrowed graph input")
	}
}
func TestTrainingWeightTapeReLUBoundary(t *testing.T) {
	// Exact zero/negative preactivation input derivatives are zero; positive seed
	// propagates unchanged through a unit final layer. Independent small integers.
	m := &LSDWeightMLP{Layers: []LinearF32{{In: 2, Out: 1, Weight: []float32{1, 0}, Bias: []float32{0}}, {In: 1, Out: 1, Weight: []float32{1}, Bias: []float32{0}}}}
	tape, err := m.BuildWeightTrainingTape([]float32{-1, 0, 0, 0, 1, 0}, []float32{2, 3, 4}, 3, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	r, err := tape.Decode(interpretFlowTape(t, tape.Tape))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "weighting ReLU values", r.Output, []float32{0, 0, 1}, 0)
	assertSliceClose(t, "weighting ReLU derivatives", r.DInput, []float32{0, 0, 0, 0, 4, 0}, 0)
}
func TestTrainingWeightTapeAdmission(t *testing.T) {
	m := tinyLSDWeighting()
	input, seed := make([]float32, 6), make([]float32, 3)
	for _, row := range []int{0, 1025} {
		if _, err := m.BuildWeightTrainingTape(input, seed, row, 1<<20); err == nil {
			t.Fatal("row")
		}
	}
	if _, err := m.BuildWeightTrainingTape(input, seed, 3, 1); err == nil {
		t.Fatal("budget")
	}
	if _, err := m.BuildWeightTrainingTape(nil, seed, 3, 1<<20); err == nil {
		t.Fatal("input")
	}
	input[0] = float32(math.NaN())
	if _, err := m.BuildWeightTrainingTape(input, seed, 3, 1<<20); err == nil {
		t.Fatal("nonfinite")
	}
	input[0] = 0
	tape, err := m.BuildWeightTrainingTape(input, seed, 3, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tape.Decode(nil); err == nil {
		t.Fatal("count")
	}
	values := interpretFlowTape(t, tape.Tape)
	values[0][0] = float32(math.Inf(1))
	if _, err := tape.Decode(values); err == nil {
		t.Fatal("output")
	}
	var zero WeightTrainingTape
	if _, err := zero.Decode(nil); err == nil {
		t.Fatal("empty")
	}
}

func TestTrainingWeightTapeTopologyAndOwnership(t *testing.T) {
	input, seed := make([]float32, 6), make([]float32, 3)
	for _, mutate := range []func(*LSDWeightMLP){func(m *LSDWeightMLP) { m.Layers[0].Bias = nil }, func(m *LSDWeightMLP) { m.Layers[1].In++ }, func(m *LSDWeightMLP) { m.Layers[0].Weight[0] = float32(math.NaN()) }, func(m *LSDWeightMLP) { m.Layers[0].Out = 16385 }} {
		m := tinyLSDWeighting()
		mutate(m)
		if _, err := m.BuildWeightTrainingTape(input, seed, 3, 1<<20); err == nil {
			t.Fatal("invalid weighting topology")
		}
	}
	m := tinyLSDWeighting()
	tape, err := m.BuildWeightTrainingTape(input, seed, 3, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	values := interpretFlowTape(t, tape.Tape)
	r, err := tape.Decode(values)
	if err != nil {
		t.Fatal(err)
	}
	values[0][0] = 42
	if r.Output[0] == 42 {
		t.Fatal("borrowed decode")
	}
	tape.Tape.Outputs[0] = -1
	if _, err := tape.Decode(values); err == nil {
		t.Fatal("bad output id")
	}
	boundary := WeightTrainingResult{Output: make([]float32, 3), DInput: make([]float32, 6)}
	if validateWeightBoundary(boundary, 3, input, seed) != nil {
		t.Fatal("valid boundary")
	}
	if validateWeightBoundary(WeightTrainingResult{}, 3) == nil {
		t.Fatal("bad boundary")
	}
	boundary.DInput = input
	if validateWeightBoundary(boundary, 3, input) == nil {
		t.Fatal("borrowed input")
	}
	boundary.Output = input[:3]
	boundary.DInput = input
	if validateWeightBoundary(boundary, 3) == nil {
		t.Fatal("overlapping output")
	}
}
