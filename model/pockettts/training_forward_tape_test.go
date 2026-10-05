package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func forwardFixture() DistillForwardInputs {
	return DistillForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}
}
func TestTrainingForwardTapeComposition(t *testing.T) {
	for _, s := range []float32{0, .3, 1} {
		for _, tt := range []float32{s, 1} {
			m := tinyTrainableFlowHead()
			in := forwardFixture()
			in.S, in.T = s, tt
			ip, err := BuildBranchTrainingTape("interpolate", [][]float32{in.Target, in.Noise}, s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			iv := interpretFlowTape(t, ip)
			fp, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{s, tt}, iv[0], 1, make([]float32, 2), make([]float32, 2), 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			fv := interpretFlowTape(t, fp.Tape)
			ap, err := BuildBranchTrainingTape("advance", [][]float32{iv[0], fv[0], fv[1]}, tt-s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			av := interpretFlowTape(t, ap)
			ep, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{tt, tt}, av[0], -1, make([]float32, 2), make([]float32, 2), 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			ev := interpretFlowTape(t, ep.Tape)
			p, err := m.BuildDistillForwardTape(in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, p.Tape)
			for i, x := range [][]float32{iv[0], fv[0], fv[1], av[0], av[1], ev[0]} {
				assertSliceClose(t, "forward composition", got[i], x, 5e-6)
			}
			if len(p.ResidentBindings) != len(fp.ResidentBindings) {
				t.Fatal("inventory")
			}
			if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingForwardTapeIndependentCPU(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := forwardFixture()
	xs := make([]float32, 2)
	for i := range xs {
		xs[i] = in.S*in.Target[i] + (1-in.S)*in.Noise[i]
	}
	v, tangent, err := m.ForwardTimeJVP(in.Condition, []float32{in.S, in.T}, xs, 1)
	if err != nil {
		t.Fatal(err)
	}
	xt, desired := make([]float32, 2), make([]float32, 2)
	delta := in.T - in.S
	for i := range xt {
		xt[i] = xs[i] + delta*v[i]
		desired[i] = v[i] + delta*tangent[i]
	}
	_, endpoint, err := m.forwardTraining(in.Condition, []float32{in.T, in.T}, xt)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildDistillForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	got := interpretFlowTape(t, p.Tape)
	for i, x := range [][]float32{xs, v, tangent, xt, desired, endpoint} {
		assertSliceClose(t, "CPU forward/JVP", got[i], x, 5e-6)
	}
}
func TestTrainingForwardTapeAdmissionOwnership(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := forwardFixture()
	p, err := m.BuildDistillForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildDistillForwardTape(in, bytes-1); err == nil {
		t.Fatal("combined source bytes")
	}
	if _, err := m.BuildDistillForwardTape(in, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"model", "time", "condition", "target", "noise", "budget"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			in := forwardFixture()
			budget := int64(64 << 20)
			switch kind {
			case "model":
				m = nil
			case "time":
				in.S = in.T + 1
			case "condition":
				in.Condition = nil
			case "target":
				in.Target = nil
			case "noise":
				in.Noise[0] = float32(math.NaN())
			case "budget":
				budget = 0
			}
			if _, err := m.BuildDistillForwardTape(in, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, x := range [][]float32{in.Condition, in.Target, in.Noise, m.Input.Weight} {
		x[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	m = tinyTrainableFlowHead()
	in = forwardFixture()
	priv, err := m.buildDistillForwardTape(in, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if !f32SlicesOverlap(priv.Tape.Tensors[priv.ResidentBindings["flow_lm.flow_net.input_proj.weight"].Parameter].Values, m.Input.Weight) {
		t.Fatal("parameter borrow")
	}
	for _, s := range priv.Tape.Tensors {
		for _, x := range [][]float32{in.Condition, in.Target, in.Noise} {
			if f32SlicesOverlap(s.Values, x) {
				t.Fatal("borrowed caller")
			}
		}
	}
	priv.Tape.Tensors = append(priv.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if err := train.ValidateResidentTapeBindings(priv.Tape, priv.ResidentBindings, 64<<20); err == nil {
		t.Fatal("dead source")
	}
}
func TestTrainingForwardResultGuards(t *testing.T) {
	in := forwardFixture()
	fresh := func() DistillForwardResult {
		return DistillForwardResult{make([]float32, 2), make([]float32, 2), make([]float32, 2), make([]float32, 2), make([]float32, 2), make([]float32, 2)}
	}
	if err := validateDistillForwardResult(fresh(), in); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "overlap", "source"} {
		r := fresh()
		switch kind {
		case "shape":
			r.Velocity = nil
		case "nan":
			r.Tangent[0] = float32(math.NaN())
		case "overlap":
			r.Desired = r.Prediction
		case "source":
			r.PrimaryInput = in.Target
		}
		if err := validateDistillForwardResult(r, in); err == nil {
			t.Fatal(kind)
		}
	}
}
