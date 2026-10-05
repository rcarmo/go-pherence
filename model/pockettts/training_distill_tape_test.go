package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func distillFixture(t *testing.T, m *FlowHeadCPU, s, tt float32) DistillTrainingInputs {
	t.Helper()
	c, x := []float32{.2, -.3, .5}, []float32{.3, -.2}
	zero := make([]float32, len(x))
	p, err := m.BuildFlowDeviceTimeTrainingTape(c, []float32{s, tt}, x, 1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	advance, err := BuildBranchTrainingTape("advance", [][]float32{x, v[0], v[1]}, tt-s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	a := interpretFlowTape(t, advance)
	e, err := m.BuildFlowDeviceTimeTrainingTape(c, []float32{tt, tt}, a[0], -1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	ev := interpretFlowTape(t, e.Tape)
	return DistillTrainingInputs{Condition: c, PrimaryInput: x, EndpointInput: a[0], Prediction: ev[0], Desired: a[1], Velocity: v[0], Tangent: v[1], Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: s, T: tt, LogVariance: .2}
}
func TestTrainingDistillTapeComposition(t *testing.T) {
	for _, s := range []float32{0, .3, 1} {
		for _, tt := range []float32{s, 1} {
			m := tinyTrainableFlowHead()
			in := distillFixture(t, m, s, tt)
			ep, err := m.BuildEndpointTrainingTape(in.Condition, tt, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.LogVariance, tt-s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			ev := interpretFlowTape(t, ep.Tape)
			pp, err := m.BuildPrimaryTrainingTape(PrimaryTrainingInputs{Condition: in.Condition, Input: in.PrimaryInput, GV: ev[1], GT: ev[2], EndpointInput: ev[0], Target: in.Target, Noise: in.Noise, S: s, T: tt, DDelta: ev[4][0]}, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			pv := interpretFlowTape(t, pp.Tape)
			p, err := m.BuildDistillTrainingTape(in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, p.Tape)
			for i := 0; i < 4; i++ {
				assertSliceClose(t, "composed distill", got[i], pv[i], 2e-5)
			}
			assertSliceClose(t, "raw stats", got[4], ev[3], 2e-6)
			source := map[int][]float32{}
			for i, id := range pp.Tape.Outputs {
				source[id] = pv[i]
			}
			for i, id := range p.Tape.Outputs {
				for name, b := range p.ResidentBindings {
					if b.Gradient == id {
						assertSliceClose(t, name, got[i], source[pp.ResidentBindings[name].Gradient], 2e-5)
					}
				}
			}
			// One live parameter identity for endpoint and primary, ONLY primary gradient.
			if len(p.ResidentBindings) != len(pp.ResidentBindings) {
				t.Fatal("inventory")
			}
			for name, b := range p.ResidentBindings {
				if b.Parameter != ep.ResidentBindings[name].Parameter || b.Gradient != len(ep.Tape.Tensors)+pp.ResidentBindings[name].Gradient {
					t.Fatal("endpoint gradient included", name)
				}
			}
			if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingDistillTapeIndependentCPU(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := distillFixture(t, m, .3, .8)
	tape, pred, err := m.forwardTraining(in.Condition, []float32{in.T, in.T}, in.EndpointInput)
	if err != nil {
		t.Fatal(err)
	}
	dr, de := make([]float32, 2), make([]float32, 2)
	square := float64(0)
	for i, p := range pred {
		d := float64(in.Desired[i]) - float64(p)
		square += d * d
		dr[i] = float32(2 * d * math.Exp(float64(in.LogVariance)) / 2)
		de[i] = -dr[i]
	}
	_, _, dx, err := m.backwardTraining(tape, de, newFlowHeadGradients(m))
	if err != nil {
		t.Fatal(err)
	}
	gv, gt := make([]float32, 2), make([]float32, 2)
	delta := in.T - in.S
	dd := float32(0)
	for i := range gv {
		gv[i] = dr[i] + delta*dx[i]
		gt[i] = delta * dr[i]
		dd += dr[i]*in.Tangent[i] + dx[i]*in.Velocity[i]
	}
	_, _, g, dc, dt, dp, err := m.ForwardTimeJVPBackward(in.Condition, []float32{in.S, in.T}, in.PrimaryInput, 1, gv, gt)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildDistillTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	target, noise := make([]float32, 2), make([]float32, 2)
	ds := float32(0)
	for i := range target {
		d := dp[i] + dx[i]
		target[i] = in.S * d
		noise[i] = (1 - in.S) * d
		ds += d * (in.Target[i] - in.Noise[i])
	}
	ds += dt[0] - dd
	assertSliceClose(t, "CPU condition", v[0], dc, 2e-5)
	assertSliceClose(t, "CPU target", v[1], target, 2e-5)
	assertSliceClose(t, "CPU noise", v[2], noise, 2e-5)
	assertSliceClose(t, "CPU times", v[3], []float32{ds, dt[1] + dd}, 2e-5)
	wants := []float64{square, math.Exp(float64(in.LogVariance))*square/2 - float64(in.LogVariance), math.Exp(float64(in.LogVariance))*square/2 - 1}
	for i, w := range wants {
		if math.Abs(float64(v[4][i])-w) > 2e-6+2e-6*math.Abs(w) {
			t.Fatal("F64 stats", i)
		}
	}
	named := flowHeadGradientMap(g)
	for i, id := range p.Tape.Outputs {
		for name, b := range p.ResidentBindings {
			if b.Gradient == id {
				assertSliceClose(t, name, v[i], named[name[len("flow_lm.flow_net."):]], 3e-5)
			}
		}
	}
}
func TestTrainingDistillTapeAdmissionOwnership(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := distillFixture(t, m, .3, .8)
	p, err := m.BuildDistillTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildDistillTrainingTape(in, bytes-1); err == nil {
		t.Fatal("combined source ceiling")
	}
	if _, err := m.BuildDistillTrainingTape(in, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"time", "prediction", "target", "seedshape", "logvar", "budget"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			in := distillFixture(t, m, .3, .8)
			budget := int64(64 << 20)
			switch kind {
			case "time":
				in.S = 1
			case "prediction":
				in.Prediction[0] = math.MaxFloat32
			case "target":
				in.Target = nil
			case "seedshape":
				in.PrimaryInput = nil
			case "logvar":
				in.LogVariance = 81
			case "budget":
				budget = 0
			}
			if _, err := m.BuildDistillTrainingTape(in, budget); err == nil {
				t.Fatal("invalid source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, x := range [][]float32{in.Condition, in.PrimaryInput, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.Target, in.Noise, m.Input.Weight} {
		x[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	m = tinyTrainableFlowHead()
	in = distillFixture(t, m, .3, .8)
	priv, err := m.buildDistillTrainingTape(in, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if !f32SlicesOverlap(priv.Tape.Tensors[priv.ResidentBindings["flow_lm.flow_net.input_proj.weight"].Parameter].Values, m.Input.Weight) {
		t.Fatal("parameter borrow")
	}
	for _, s := range priv.Tape.Tensors {
		for _, x := range [][]float32{in.Condition, in.PrimaryInput, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.Target, in.Noise} {
			if f32SlicesOverlap(s.Values, x) {
				t.Fatal("input borrowed")
			}
		}
	}
	// Source admission includes dead endpoint gradients/placeholders. Invalid
	// dead constants must fail before pruning can hide them.
	priv.Tape.Tensors = append(priv.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if err := train.ValidateResidentTapeBindings(priv.Tape, priv.ResidentBindings, 64<<20); err == nil {
		t.Fatal("dead source admitted")
	}
}
func TestTrainingDistillResultGuards(t *testing.T) {
	in := DistillTrainingInputs{Condition: make([]float32, 3), PrimaryInput: make([]float32, 2), Prediction: make([]float32, 2)}
	fresh := func() DistillTrainingResult {
		return DistillTrainingResult{PrimaryTrainingResult{make([]float32, 3), make([]float32, 2), make([]float32, 2), make([]float32, 2)}, 1, 0, 0}
	}
	if err := validateDistillResult(fresh(), in); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"stats", "shape", "borrowed"} {
		r := fresh()
		switch kind {
		case "stats":
			r.RawSquare = -1
		case "shape":
			r.DTimes = nil
		case "borrowed":
			r.DTarget = in.Prediction
		}
		if err := validateDistillResult(r, in); err == nil {
			t.Fatal(kind)
		}
	}
}
