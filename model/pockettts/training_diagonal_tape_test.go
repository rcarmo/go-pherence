package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingDiagonalTapeComposition(t *testing.T) {
	for _, time := range []float32{0, .3, 1} {
		for _, scale := range []float32{0, .25, 1} {
			for _, lv := range []float32{-2, 0, 2} {
				o := loadTrainingStepOracle(t)
				_, m, _ := trainingStepModelsFromOracle(t, o)
				condition := append([]float32(nil), make([]float32, m.Condition.In)...)
				input := append([]float32(nil), o.NormalizedLatents[:m.Input.In]...)
				desired := make([]float32, m.Input.In)
				for i := range desired {
					desired[i] = float32(i%7-3) * .1
				}
				forward, err := m.BuildFlowDeviceTimeTrainingTape(condition, []float32{time, time}, input, -1, make([]float32, len(input)), make([]float32, len(input)), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				fv := interpretFlowTape(t, forward.Tape)
				loss, err := BuildLSDLossTrainingTape(fv[0], desired, lv, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				values := interpretFlowTape(t, loss)
				seed := append([]float32(nil), values[1]...)
				for i := range seed {
					seed[i] *= scale
				}
				back, err := m.BuildFlowDeviceTimeTrainingTape(condition, []float32{time, time}, input, -1, seed, make([]float32, len(input)), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				bv := interpretFlowTape(t, back.Tape)
				target, err := BuildBranchTrainingTape("diagonal-target", [][]float32{bv[3], seed}, time, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				tv := interpretFlowTape(t, target)
				p, err := m.BuildDiagonalTrainingTape(condition, time, input, fv[0], desired, lv, scale, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, p.Tape)
				assertSliceClose(t, "condition", got[0], bv[2], 2e-5)
				assertSliceClose(t, "target", got[1], tv[0], 2e-5)
				assertSliceClose(t, "raw stats", got[2], values[0], 2e-6)
				want := map[string][]float32{}
				for i, id := range back.Tape.Outputs {
					for name, b := range back.ResidentBindings {
						if b.Gradient == id {
							want[name] = bv[i]
						}
					}
				}
				v := map[int][]float32{}
				for i, id := range p.Tape.Outputs {
					v[id] = got[i]
				}
				if len(p.ResidentBindings) != len(back.ResidentBindings) {
					t.Fatal("inventory")
				}
				for name, b := range p.ResidentBindings {
					assertSliceClose(t, name, v[b.Gradient], want[name], 2e-5)
				}
				if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
func TestTrainingDiagonalTapeIndependentCPU(t *testing.T) {
	o := loadTrainingStepOracle(t)
	_, m, _ := trainingStepModelsFromOracle(t, o)
	condition := make([]float32, m.Condition.In)
	input := o.NormalizedLatents[:m.Input.In]
	desired := make([]float32, len(input))
	time, lv, scale := float32(.3), float32(.2), float32(.25)
	// Independent ordinary CPU flow and scalar F64 loss seed.
	tape, prediction, err := m.forwardTraining(condition, []float32{time, time}, input)
	if err != nil {
		t.Fatal(err)
	}
	seed := make([]float32, len(input))
	square := float64(0)
	for i, p := range prediction {
		d := float64(p) - float64(desired[i])
		square += d * d
		seed[i] = float32(2*d*math.Exp(float64(lv))/float64(len(input))) * scale
	}
	grads := newFlowHeadGradients(m)
	dc, _, dx, err := m.backwardTraining(tape, seed, grads)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildDiagonalTrainingTape(condition, time, input, prediction, desired, lv, scale, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	assertSliceClose(t, "CPU condition", v[0], dc, 5e-5)
	target := make([]float32, len(input))
	for i := range target {
		target[i] = time*dx[i] - seed[i]
	}
	assertSliceClose(t, "CPU target", v[1], target, 5e-5)
	want := []float64{square, math.Exp(float64(lv))*square/float64(len(input)) - float64(lv), math.Exp(float64(lv))*square/float64(len(input)) - 1}
	for i, x := range want {
		if math.Abs(float64(v[2][i])-x) > 2e-6+2e-6*math.Abs(x) {
			t.Fatal("F64 loss", i, v[2][i], x)
		}
	}
	named := flowHeadGradientMap(grads)
	for i, id := range p.Tape.Outputs {
		for name, b := range p.ResidentBindings {
			if id == b.Gradient {
				assertSliceClose(t, name, v[i], named[name[len("flow_lm.flow_net."):]], 5e-5)
			}
		}
	}
}
func TestTrainingDiagonalTapeAdmissionOwnership(t *testing.T) {
	o := loadTrainingStepOracle(t)
	_, m, _ := trainingStepModelsFromOracle(t, o)
	c := make([]float32, m.Condition.In)
	x := o.NormalizedLatents[:m.Input.In]
	prediction, desired := make([]float32, len(x)), make([]float32, len(x))
	p, err := m.BuildDiagonalTrainingTape(c, .3, x, prediction, desired, 0, .25, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err = m.BuildDiagonalTrainingTape(c, .3, x, prediction, desired, 0, .25, bytes-1); err == nil {
		t.Fatal("source byte ceiling")
	}
	if _, err = m.BuildDiagonalTrainingTape(c, .3, x, prediction, desired, 0, .25, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"time", "scale", "prediction", "logvar", "condition", "shape", "budget"} {
		t.Run(kind, func(t *testing.T) {
			time, scale, lv := float32(.3), float32(.25), float32(0)
			pr := append([]float32(nil), prediction...)
			cc := append([]float32(nil), c...)
			budget := int64(64 << 20)
			switch kind {
			case "time":
				time = -1
			case "scale":
				scale = 2
			case "prediction":
				pr[0] = math.MaxFloat32
			case "logvar":
				lv = 81
			case "condition":
				cc[0] = float32(math.NaN())
			case "shape":
				pr = nil
			case "budget":
				budget = 0
			}
			if _, err := m.BuildDiagonalTrainingTape(cc, time, x, pr, desired, lv, scale, budget); err == nil {
				t.Fatal("bad input")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	m.Input.Weight[0] = 99
	c[0] = 99
	x[0] = 99
	prediction[0] = 99
	desired[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, v := range before {
		assertSliceClose(t, "public owned", after[i], v, 0)
	}
}
func TestTrainingDiagonalResultGuards(t *testing.T) {
	c, x := make([]float32, 3), make([]float32, 2)
	fresh := func() DiagonalTrainingResult {
		return DiagonalTrainingResult{make([]float32, 3), make([]float32, 2), 1, 0, 0}
	}
	if err := validateDiagonalResult(fresh(), c, x, x, x); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "square", "overlap", "source"} {
		r := fresh()
		switch kind {
		case "shape":
			r.DTarget = nil
		case "nan":
			r.Loss = math.NaN()
		case "square":
			r.RawSquare = -1
		case "overlap":
			r.DTarget = r.DCondition[:2]
		case "source":
			r.DTarget = x
		}
		if err := validateDiagonalResult(r, c, x, x, x); err == nil {
			t.Fatal(kind)
		}
	}
}
