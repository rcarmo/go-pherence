package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingEOSProjectionTapeReference(t *testing.T) {
	m := &FlowLMTrainingCPU{Hidden: 3, EOS: LinearF32{In: 3, Out: 1, Weight: []float32{.4, -.7, .2}, Bias: []float32{-.1}}}
	input := []float32{.2, .8, -.3, .7, -.2, .6, -.4, .3, .9, 2, 3, 4}
	mask := []bool{true, true, false, false}
	for _, weight := range []float32{0, .1, 1, 2} {
		p, err := m.BuildEOSProjectionTrainingTape(input, mask, weight, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		if err = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 1<<20); err != nil {
			t.Fatal(err)
		}
		out := interpretFlowTape(t, p.Tape)
		loss := 0.0
		dx, dw := make([]float64, len(input)), make([]float64, 3)
		db := 0.0
		for row := 0; row < 3; row++ {
			logit := float64(m.EOS.Bias[0])
			for k, w := range m.EOS.Weight {
				logit += float64(w) * float64(input[row*3+k])
			}
			label := 0.0
			if row == 2 {
				label = 1
			}
			loss += (math.Max(logit, 0) - logit*label + math.Log1p(math.Exp(-math.Abs(logit)))) / 3
			seed := (1/(1+math.Exp(-logit)) - label) * float64(weight) / 3
			db += seed
			for k, w := range m.EOS.Weight {
				dx[row*3+k] = seed * float64(w)
				dw[k] += seed * float64(input[row*3+k])
			}
		}
		if math.Abs(float64(out[0][0])-loss) > 2e-7 {
			t.Fatal("independent BCE mean", out[0][0], loss)
		}
		check := func(name string, x []float32, y []float64) {
			t.Helper()
			if len(x) != len(y) {
				t.Fatal(name, "shape")
			}
			for i, v := range x {
				if math.Abs(float64(v)-y[i]) > 2e-7+2e-6*math.Abs(y[i]) {
					t.Fatal(name, i, v, y[i])
				}
			}
		}
		check("hidden VJP", out[1], dx)
		check("weight VJP", out[2], dw)
		check("bias VJP", out[3], []float64{db})
		for _, v := range out[1][9:] {
			if v != 0 {
				t.Fatal("inactive hidden VJP")
			}
		}
		// Independently compose the old projection/loss/projection boundary path.
		f, err := BuildProjectionTrainingTape(m.EOS, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", input, make([]float32, 4), 4, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		fv := interpretFlowTape(t, f.Tape)
		l, err := BuildEOSTrainingTape(fv[0], mask, weight, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		lv := interpretFlowTape(t, l)
		b, err := BuildProjectionTrainingTape(m.EOS, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", input, lv[1], 4, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		bv := interpretFlowTape(t, b.Tape)
		assertSliceClose(t, "legacy mean", out[0], lv[0], 2e-7)
		for i := 1; i < 4; i++ {
			assertSliceClose(t, "composed legacy VJP", out[i], bv[i], 2e-7)
		}
	}
}
func TestTrainingEOSProjectionTapeAdmission(t *testing.T) {
	m := &FlowLMTrainingCPU{Hidden: 2, EOS: LinearF32{In: 2, Out: 1, Weight: []float32{.4, -.7}, Bias: []float32{-.1}}}
	for _, mask := range [][]bool{nil, {false, false}, {true, false, true}} {
		if _, err := m.BuildEOSProjectionTrainingTape(make([]float32, len(mask)*2), mask, .1, 1<<20); err == nil {
			t.Fatal("mask", mask)
		}
	}
	if _, err := m.BuildEOSProjectionTrainingTape([]float32{.1, .2}, []bool{true}, .1, 1); err == nil {
		t.Fatal("budget")
	}
	if _, err := m.BuildEOSProjectionTrainingTape([]float32{float32(math.NaN()), .2}, []bool{true}, .1, 1<<20); err == nil {
		t.Fatal("nonfinite")
	}
	for _, w := range []float32{-1, float32(math.NaN())} {
		if _, err := m.BuildEOSProjectionTrainingTape([]float32{.1, .2}, []bool{true}, w, 1<<20); err == nil {
			t.Fatal("weight")
		}
	}
	m.EOS.Weight[0] = math.MaxFloat32
	if _, err := m.BuildEOSProjectionTrainingTape([]float32{2, .2}, []bool{true}, .1, 1<<20); err == nil {
		t.Fatal("overflow")
	}
	m.EOS.Weight[0] = .4
	input := []float32{.1, .2}
	p, err := m.BuildEOSProjectionTrainingTape(input, []bool{true}, .1, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	before := interpretFlowTape(t, p.Tape)
	input[0] = 99
	m.EOS.Weight[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	if validateEOSProjectionBoundary(EOSProjectionTrainingResult{Loss: 1, DInput: input}, 1, 2, input) == nil {
		t.Fatal("borrowed boundary")
	}
}
