package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func primaryFixture() PrimaryTrainingInputs {
	return PrimaryTrainingInputs{Condition: []float32{.2, -.3, .5}, Input: []float32{.3, -.2}, GV: []float32{.2, .1}, GT: []float32{-.1, .2}, EndpointInput: []float32{.05, -.02}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .25, T: .8, DDelta: .03}
}
func TestTrainingPrimaryTapeComposition(t *testing.T) {
	for _, s := range []float32{0, .3, 1} {
		for _, tt := range []float32{s, 1} {
			for _, zero := range []bool{false, true} {
				m := tinyTrainableFlowHead()
				in := primaryFixture()
				in.S, in.T = s, tt
				if zero {
					clear(in.GV)
					clear(in.GT)
					clear(in.EndpointInput)
					in.DDelta = 0
				}
				flow, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{s, tt}, in.Input, 1, in.GV, in.GT, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				fv := interpretFlowTape(t, flow.Tape)
				finish, err := BuildBranchTrainingTape("finish", [][]float32{fv[3], in.EndpointInput, in.Target, in.Noise, {fv[4][0], fv[5][0]}, {in.DDelta}}, s, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				v := interpretFlowTape(t, finish)
				p, err := m.BuildPrimaryTrainingTape(in, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, p.Tape)
				for i, x := range [][]float32{fv[2], v[0], v[1], v[2]} {
					assertSliceClose(t, "primary composed", got[i], x, 2e-5)
				}
				values := map[int][]float32{}
				for i, id := range flow.Tape.Outputs {
					values[id] = fv[i]
				}
				for i, id := range p.Tape.Outputs {
					for name, b := range p.ResidentBindings {
						if b.Gradient == id {
							assertSliceClose(t, name, got[i], values[flow.ResidentBindings[name].Gradient], 2e-5)
						}
					}
				}
				if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
func TestTrainingPrimaryTapeIndependentCPU(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	_, _, g, dc, dt, dx, err := m.ForwardTimeJVPBackward(in.Condition, []float32{in.S, in.T}, in.Input, 1, in.GV, in.GT)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildPrimaryTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	target, noise := make([]float32, len(dx)), make([]float32, len(dx))
	ds := float64(0)
	for i, d := range dx {
		total := d + in.EndpointInput[i]
		target[i] = in.S * total
		noise[i] = (1 - in.S) * total
		ds += float64(total) * float64(in.Target[i]-in.Noise[i])
	}
	assertSliceClose(t, "CPU condition", v[0], dc, 2e-5)
	assertSliceClose(t, "CPU target", v[1], target, 2e-5)
	assertSliceClose(t, "CPU noise", v[2], noise, 2e-5)
	want := []float64{ds + float64(dt[0]) - float64(in.DDelta), float64(dt[1]) + float64(in.DDelta)}
	for i, w := range want {
		if math.Abs(float64(v[3][i])-w) > 2e-5+2e-5*math.Abs(w) {
			t.Fatal("CPU/F64 times", i, v[3][i], w)
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
func TestTrainingPrimaryTapeAdmissionOwnership(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	p, err := m.BuildPrimaryTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildPrimaryTrainingTape(in, bytes-1); err == nil {
		t.Fatal("full bytes")
	}
	if _, err := m.BuildPrimaryTrainingTape(in, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"model", "time", "delta", "endpoint", "noise", "seed", "condition", "budget"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			in := primaryFixture()
			budget := int64(64 << 20)
			switch kind {
			case "model":
				m = nil
			case "time":
				in.S = in.T + 1
			case "delta":
				in.DDelta = float32(math.NaN())
			case "endpoint":
				in.EndpointInput = nil
			case "noise":
				in.Noise[0] = float32(math.NaN())
			case "seed":
				in.GT = nil
			case "condition":
				in.Condition = nil
			case "budget":
				budget = 0
			}
			if _, err := m.BuildPrimaryTrainingTape(in, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, x := range [][]float32{in.Condition, in.Input, in.GV, in.GT, in.EndpointInput, in.Target, in.Noise, m.Input.Weight} {
		x[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	// Private borrowing is restricted to named parameters.
	m = tinyTrainableFlowHead()
	in = primaryFixture()
	private, err := m.buildPrimaryTrainingTape(in, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if !f32SlicesOverlap(private.Tape.Tensors[private.ResidentBindings["flow_lm.flow_net.input_proj.weight"].Parameter].Values, m.Input.Weight) {
		t.Fatal("parameter not borrowed")
	}
	for _, s := range private.Tape.Tensors {
		for _, x := range [][]float32{in.Condition, in.Input, in.GV, in.GT, in.EndpointInput, in.Target, in.Noise} {
			if f32SlicesOverlap(s.Values, x) {
				t.Fatal("borrowed nonparam")
			}
		}
	}
}
func TestTrainingPrimaryResultGuards(t *testing.T) {
	in := primaryFixture()
	fresh := func() PrimaryTrainingResult {
		return PrimaryTrainingResult{make([]float32, 3), make([]float32, 2), make([]float32, 2), make([]float32, 2)}
	}
	if err := validatePrimaryResult(fresh(), in); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "overlap", "source"} {
		r := fresh()
		switch kind {
		case "shape":
			r.DTimes = nil
		case "nan":
			r.DNoise[0] = float32(math.NaN())
		case "overlap":
			r.DTarget = r.DNoise
		case "source":
			r.DTarget = in.Target
		}
		if err := validatePrimaryResult(r, in); err == nil {
			t.Fatal(kind)
		}
	}
}
