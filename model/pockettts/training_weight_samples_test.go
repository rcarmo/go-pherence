package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

// Independent scalar F64 affine/ReLU VJP. Samples are indexed directly rather
// than interpreting the device packing graph or using its assembled matrix.
func sampleWeightOracle(m *LSDWeightMLP, s WeightSampleInputs) ([]float64, []float64, map[string][]float64) {
	frames := len(s.DiagonalTime)
	output := make([]float64, 2*frames)
	dx := make([]float64, 4*frames)
	params := map[string][]float64{}
	for i, l := range m.Layers {
		prefix := weightSampleParameterPrefix(i)
		params[prefix+".weight"] = make([]float64, len(l.Weight))
		params[prefix+".bias"] = make([]float64, len(l.Bias))
	}
	for row := 0; row < 2*frames; row++ {
		index := row % frames
		a, b, seed := s.DiagonalTime[index], s.DiagonalTime[index], s.DiagonalSeed[index]
		if row >= frames {
			a, b, seed = s.DistillS[index], s.DistillT[index], s.DistillSeed[index]
		}
		x := []float64{float64(a), float64(b)}
		inputs := make([][]float64, len(m.Layers))
		pre := make([][]float64, len(m.Layers))
		for i, l := range m.Layers {
			inputs[i] = x
			z := make([]float64, l.Out)
			for o := range z {
				z[o] = float64(l.Bias[o])
				for k := 0; k < l.In; k++ {
					z[o] += x[k] * float64(l.Weight[o*l.In+k])
				}
			}
			pre[i] = z
			x = append([]float64(nil), z...)
			if i < len(m.Layers)-1 {
				for k := range x {
					x[k] = math.Max(0, x[k])
				}
			}
		}
		output[row] = x[0]
		dy := []float64{float64(seed)}
		for i := len(m.Layers) - 1; i >= 0; i-- {
			l := m.Layers[i]
			if i < len(m.Layers)-1 {
				for k := range dy {
					if pre[i][k] <= 0 {
						dy[k] = 0
					}
				}
			}
			prefix := weightSampleParameterPrefix(i)
			next := make([]float64, l.In)
			for o, v := range dy {
				params[prefix+".bias"][o] += v
				for k := 0; k < l.In; k++ {
					params[prefix+".weight"][o*l.In+k] += inputs[i][k] * v
					next[k] += float64(l.Weight[o*l.In+k]) * v
				}
			}
			dy = next
		}
		dx[2*row], dx[2*row+1] = dy[0], dy[1]
	}
	return output, dx, params
}
func weightSampleParameterPrefix(i int) string { // Test-only independent naming.
	const digits = "0123456789"
	return "flow.w_s_t." + string(digits[2*i])
}
func TestTrainingSampleWeightTapeReference(t *testing.T) {
	m := weightTapeFixture()
	for _, frames := range []int{1, 3, 257, 512} {
		s := WeightSampleInputs{}
		for _, dst := range []*[]float32{&s.DiagonalTime, &s.DistillS, &s.DistillT, &s.DiagonalSeed, &s.DistillSeed} {
			*dst = make([]float32, frames)
		}
		for i := 0; i < frames; i++ {
			s.DiagonalTime[i] = (float32(i%11) + .37) / 11
			s.DistillS[i] = (float32(i%7) + .23) / 10
			s.DistillT[i] = s.DistillS[i] + .2
			s.DiagonalSeed[i] = float32(i%5-2) / float32(frames)
			s.DistillSeed[i] = float32(i%3-1) / float32(frames)
		}
		tape, err := m.BuildSampleWeightTrainingTape(s, 1<<24)
		if err != nil {
			t.Fatal(frames, err)
		}
		if err = tape.Tape.Validate(1 << 24); err != nil {
			t.Fatal(err)
		}
		if err = train.ValidateResidentTapeBindings(tape.Tape, tape.ResidentBindings, 1<<24); err != nil {
			t.Fatal(err)
		}
		got, err := tape.Decode(interpretFlowTape(t, tape.Tape))
		if err != nil {
			t.Fatal(err)
		}
		values, dx, params := sampleWeightOracle(m, s)
		check := func(name string, x []float32, y []float64) {
			t.Helper()
			if len(x) != len(y) {
				t.Fatal(name, "shape")
			}
			for i, v := range x {
				if e := math.Abs(float64(v) - y[i]); e > 2e-7+2e-6*math.Abs(y[i]) {
					t.Fatal(frames, name, i, v, y[i], e)
				}
			}
		}
		check("output", got.Output, values)
		check("MLP-input VJP", got.DInput, dx)
		for name, x := range got.Parameters {
			check(name, x, params[name])
		}
		// Legacy host matrix is independently packed for differential topology/order.
		input, seed := make([]float32, 4*frames), make([]float32, 2*frames)
		for i := 0; i < frames; i++ {
			input[2*i], input[2*i+1] = s.DiagonalTime[i], s.DiagonalTime[i]
			input[2*frames+2*i], input[2*frames+2*i+1] = s.DistillS[i], s.DistillT[i]
			seed[i], seed[frames+i] = s.DiagonalSeed[i], s.DistillSeed[i]
		}
		legacy, err := m.BuildWeightTrainingTape(input, seed, 2*frames, 1<<24)
		if err != nil {
			t.Fatal(err)
		}
		want, err := legacy.Decode(interpretFlowTape(t, legacy.Tape))
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "legacy output", got.Output, want.Output, 2e-7)
		assertSliceClose(t, "legacy inputVJP", got.DInput, want.DInput, 2e-6)
		for n, x := range got.Parameters {
			assertSliceClose(t, n, x, want.Parameters[n], 2e-6)
		}
		// Packing occurs inside the forward dependency closure; VJPs/seed constants
		// disappear on pure-forward pruning, but complete source admission remains.
		pruned, remap, err := train.PruneTape(tape.Tape, tape.Tape.Outputs[:1], 1<<24)
		if err != nil {
			t.Fatal(err)
		}
		out := interpretFlowTape(t, pruned)
		assertSliceClose(t, "pruned output", out[0], got.Output, 2e-7)
		if remap[tape.Tape.Outputs[1]] != -1 || len(pruned.Commands) >= len(tape.Tape.Commands) {
			t.Fatal("forward VJP not pruned")
		}
		var rawFloats int
		for id, spec := range tape.Tape.Tensors {
			isParameter := false
			for _, binding := range tape.ResidentBindings {
				if binding.Parameter == id {
					isParameter = true
				}
			}
			if !isParameter && len(spec.Values) > 0 {
				nonzero := false
				for _, v := range spec.Values {
					nonzero = nonzero || v != 0
				}
				if nonzero {
					rawFloats += len(spec.Values)
				}
			}
		}
		if frames > 1 && rawFloats != 5*frames {
			t.Fatal("raw time/seed upload inventory", frames, rawFloats)
		}
	}
}
func TestTrainingSampleWeightTapeAdmission(t *testing.T) {
	m := weightTapeFixture()
	base := WeightSampleInputs{[]float32{.25, .75}, []float32{.1, .3}, []float32{.2, .9}, []float32{.1, -.2}, []float32{.4, .3}}
	for field := 0; field < 5; field++ {
		for _, bad := range [][]float32{nil, {1}, {1, 2, 3}, {float32(math.NaN()), 0}, {0, float32(math.Inf(1))}} {
			s := base
			dst := []*[]float32{&s.DiagonalTime, &s.DistillS, &s.DistillT, &s.DiagonalSeed, &s.DistillSeed}
			*dst[field] = bad
			if _, err := m.BuildSampleWeightTrainingTape(s, 1<<20); err == nil {
				t.Fatal("column", field, bad)
			}
		}
	}
	for _, frames := range []int{0, 513} {
		s := WeightSampleInputs{DiagonalTime: make([]float32, frames)}
		if _, err := m.BuildSampleWeightTrainingTape(s, 1<<20); err == nil {
			t.Fatal("frame admission")
		}
	}
	for _, rows := range []int{1, 3, 1026} {
		if base.validate(rows) == nil {
			t.Fatal("rows", rows)
		}
	}
	if _, err := m.BuildSampleWeightTrainingTape(base, 1); err == nil {
		t.Fatal("byte ceiling")
	}
	tape, err := m.BuildSampleWeightTrainingTape(base, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float32(nil), tape.Tape.Tensors[0].Values...)
	base.DiagonalTime[0] = 99
	if !reflect.DeepEqual(before, tape.Tape.Tensors[0].Values) {
		t.Fatal("borrowed column")
	}
	// Finite padding times need not satisfy active-row timing rules here; the
	// complete step performs mask/time admission before this graph is built.
	base.DiagonalTime[0] = -10
	if _, err = m.BuildSampleWeightTrainingTape(base, 1<<20); err != nil {
		t.Fatal("finite padding", err)
	}
}

func weightTapeFixture() *LSDWeightMLP {
	return &LSDWeightMLP{Layers: []LinearF32{{In: 2, Out: 3, Weight: []float32{1, -2, .5, 1, -1, .25}, Bias: []float32{.1, -.2, .3}}, {In: 3, Out: 1, Weight: []float32{.2, -.4, .7}, Bias: []float32{-.1}}}}
}

func TestTrainingSampleWeightTapeReLUBoundary(t *testing.T) {
	m := &LSDWeightMLP{Layers: []LinearF32{{In: 2, Out: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}}, {In: 2, Out: 1, Weight: []float32{1, -1}, Bias: []float32{0}}}}
	s := WeightSampleInputs{[]float32{0, 1}, []float32{-1, 0}, []float32{0, 1}, []float32{1, 1}, []float32{1, 1}}
	plan, err := m.BuildSampleWeightTrainingTape(s, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	got, err := plan.Decode(interpretFlowTape(t, plan.Tape))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "exact ReLU output", got.Output, []float32{0, 0, 0, -1}, 0)
	assertSliceClose(t, "exact ReLU MLPinput VJP", got.DInput, []float32{0, 0, 1, -1, 0, 0, 0, -1}, 0)
}
