package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func diagForwardFixture() DiagonalForwardInputs {
	return DiagonalForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: .3}
}
func TestTrainingDiagForwardTapeReferences(t *testing.T) {
	for _, time := range []float32{0, .3, 1} {
		m := tinyTrainableFlowHead()
		in := diagForwardFixture()
		in.Time = time
		ip, err := BuildBranchTrainingTape("interpolate", [][]float32{in.Target, in.Noise}, time, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		iv := interpretFlowTape(t, ip)
		fp, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{time, time}, iv[0], -1, make([]float32, 2), make([]float32, 2), 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		fv := interpretFlowTape(t, fp.Tape)
		p, err := m.BuildDiagonalForwardTape(in, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, p.Tape)
		for i, x := range [][]float32{iv[0], iv[1], fv[0]} {
			assertSliceClose(t, "diagonal forward composition", got[i], x, 5e-6)
		}
		// Independent ordinary CPU flow and scalar interpolation/order.
		x, desired := make([]float32, 2), make([]float32, 2)
		for i := range x {
			x[i] = time*in.Target[i] + (1-time)*in.Noise[i]
			desired[i] = in.Target[i] - in.Noise[i]
		}
		_, prediction, err := m.forwardTraining(in.Condition, []float32{time, time}, x)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range [][]float32{x, desired, prediction} {
			assertSliceClose(t, "CPU diagonal forward", got[i], v, 5e-6)
		}
		if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
			t.Fatal(err)
		}
	}
}
func TestTrainingDiagForwardTapeAdmissionOwnership(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := diagForwardFixture()
	p, err := m.BuildDiagonalForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildDiagonalForwardTape(in, bytes-1); err == nil {
		t.Fatal("combined source budget")
	}
	if _, err := m.BuildDiagonalForwardTape(in, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"model", "time", "condition", "target", "noise", "budget"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			in := diagForwardFixture()
			budget := int64(64 << 20)
			switch kind {
			case "model":
				m = nil
			case "time":
				in.Time = -1
			case "condition":
				in.Condition = nil
			case "target":
				in.Target = nil
			case "noise":
				in.Noise[0] = float32(math.NaN())
			case "budget":
				budget = 0
			}
			if _, err := m.BuildDiagonalForwardTape(in, budget); err == nil {
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
	in = diagForwardFixture()
	priv, err := m.buildDiagonalForwardTape(in, 64<<20, true)
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
func TestTrainingDiagForwardResultGuards(t *testing.T) {
	in := diagForwardFixture()
	fresh := func() DiagonalForwardResult {
		return DiagonalForwardResult{make([]float32, 2), make([]float32, 2), make([]float32, 2)}
	}
	if err := validateDiagonalForwardResult(fresh(), in); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "overlap", "source"} {
		r := fresh()
		switch kind {
		case "shape":
			r.Input = nil
		case "nan":
			r.Prediction[0] = float32(math.NaN())
		case "overlap":
			r.Input = r.Desired
		case "source":
			r.Desired = in.Target
		}
		if err := validateDiagonalForwardResult(r, in); err == nil {
			t.Fatal(kind)
		}
	}
}
