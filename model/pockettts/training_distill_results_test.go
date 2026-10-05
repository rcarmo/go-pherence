package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingDistillResultsTape(t *testing.T) {
	for _, s := range []float32{0, .3, 1} {
		for _, tt := range []float32{s, 1} {
			m := tinyTrainableFlowHead()
			in := distillFixture(t, m, s, tt)
			p, err := m.BuildDistillTrainingTape(in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			full := interpretFlowTape(t, p.Tape)
			q, err := buildDistillResultsTape(p, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			all := interpretFlowTape(t, q.Tape)
			byID := map[int][]float32{}
			for i, id := range q.Tape.Outputs {
				byID[id] = all[i]
			}
			out := [][]float32{}
			for _, id := range distillResultsOutputs(q) {
				out = append(out, byID[id])
			}
			r, err := decodeDistillResults(out, in)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceClose(t, "raw condition", r.DCondition, full[0], 2e-5)
			assertSliceClose(t, "raw target", r.DTarget, full[1], 2e-5)
			assertClose64(t, "square", r.RawSquare, float64(full[4][0]), 2e-6)
			assertClose64(t, "loss", r.Loss, float64(full[4][1]), 2e-6)
			if r.DLogVariance != full[4][2] {
				t.Fatal("logvar")
			}
			for name, b := range p.ResidentBindings {
				if q.ResidentBindings[name] != b {
					t.Fatal("changed roots")
				}
			}
			var bytes int64
			for _, x := range q.Tape.Tensors {
				n, _ := x.Elements()
				bytes += 4 * int64(n)
			}
			if _, err := buildDistillResultsTape(p, bytes-1); err == nil {
				t.Fatal("witness byte ceiling")
			}
			if _, err := buildDistillResultsTape(p, bytes); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingDistillResultsAdmission(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := distillFixture(t, m, .3, .8)
	p, err := m.BuildDistillTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	bad := p
	bad.Tape.Outputs = bad.Tape.Outputs[:4]
	bad.ResidentBindings = nil
	if _, err := buildDistillResultsTape(bad, 64<<20); err == nil {
		t.Fatal("bad bindings")
	}
	// Binding gradients must stay declared, so use a valid reduced topology for
	// this malformed internal result layout, distinct from full-source admission.
	bad = p
	bad.Tape.Outputs = nil
	for _, b := range bad.ResidentBindings {
		bad.Tape.Outputs = append(bad.Tape.Outputs, b.Gradient)
	}
	bad.Tape.Outputs = bad.Tape.Outputs[:1]
	bad.ResidentBindings = map[string]train.ResidentTapeBinding{}
	for name, b := range p.ResidentBindings {
		if b.Gradient == bad.Tape.Outputs[0] {
			bad.ResidentBindings[name] = b
			break
		}
	}
	if _, err := buildDistillResultsTape(bad, 64<<20); err == nil {
		t.Fatal("bad result count")
	}
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if _, err := buildDistillResultsTape(p, 64<<20); err == nil {
		t.Fatal("dead nonfinite source")
	}
}
func TestTrainingDistillResultsBoundary(t *testing.T) {
	in := DistillTrainingInputs{Condition: make([]float32, 3), PrimaryInput: make([]float32, 2), EndpointInput: make([]float32, 2), Prediction: make([]float32, 2), Desired: make([]float32, 2), Velocity: make([]float32, 2), Tangent: make([]float32, 2), Target: make([]float32, 2), Noise: make([]float32, 2)}
	fresh := func() [][]float32 { return [][]float32{make([]float32, 3), make([]float32, 2), {1, -.2, -.1}, {0}} }
	if _, err := decodeDistillResults(fresh(), in); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "shape", "nan", "stats", "witness", "overlap", "borrowed"} {
		v := fresh()
		switch kind {
		case "count":
			v = v[:3]
		case "shape":
			v[1] = nil
		case "nan":
			v[2][1] = float32(math.NaN())
		case "stats":
			v[2][0] = -1
		case "witness":
			v[3][0] = 1
		case "overlap":
			v[1] = v[0][:2]
		case "borrowed":
			v[1] = in.EndpointInput
		}
		if _, err := decodeDistillResults(v, in); err == nil {
			t.Fatal(kind)
		}
	}
}
func TestTrainingDistillResultsFiniteWitness(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := distillFixture(t, m, .3, .8)
	p, err := m.BuildDistillTrainingTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, slot := range []int{2, 3} {
		for _, x := range []float32{0, math.MaxFloat32, -math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
			q, err := buildDistillResultsTape(p, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			id := q.Tape.Outputs[slot]
			n, _ := q.Tape.Tensors[id].Elements()
			cmds := []train.Command{}
			for _, c := range q.Tape.Commands {
				if c.Output != id {
					cmds = append(cmds, c)
				}
			}
			if isFinite(x) {
				q.Tape.Tensors[id].Values = make([]float32, n)
				for i := range q.Tape.Tensors[id].Values {
					q.Tape.Tensors[id].Values[i] = x
				}
			} else {
				g := &transformerTapeGraph{tape: q.Tape, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
				a := g.constant([]int{1, 1}, []float32{math.MaxFloat32})
				tangent := g.zero(1, 1)
				weights := make([]float32, n)
				for i := range weights {
					weights[i] = 2
					if math.IsInf(float64(x), -1) {
						weights[i] = -2
					}
				}
				w := g.constant([]int{n, 1}, weights)
				bias := g.zero(n)
				prefix := []train.Command{}
				q.Tape.Tensors[id].Shape = []int{1, n}
				g.tape.Tensors[id].Shape = []int{1, n}
				if math.IsNaN(float64(x)) {
					tmp := g.tensor([]int{1, 1})
					prefix = append(prefix, train.Command{Op: "linear-forward", Inputs: []int{a, tangent, g.constant([]int{1, 1}, []float32{2}), g.zero(1)}, Output: tmp})
					prefix = append(prefix, train.Command{Op: "linear-forward", Inputs: []int{tmp, tangent, g.zero(n, 1), bias}, Output: id})
				} else {
					prefix = append(prefix, train.Command{Op: "linear-forward", Inputs: []int{a, tangent, w, bias}, Output: id})
				}
				g.tape.Commands = append(prefix, cmds...)
				built, err := materializeSequenceTape(g, q.ResidentBindings, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				q.Tape = built.Tape
				cmds = q.Tape.Commands
			}
			q.Tape.Commands = cmds
			v := interpretFlowTape(t, q.Tape)
			witness := v[len(v)-1][0]
			if isFinite(x) {
				if witness != 0 {
					t.Fatal("finite derivative overflow", slot, x, witness)
				}
			} else if isFinite(witness) {
				t.Fatal("invalid derivative hidden", slot, x, witness)
			}
		}
	}
}
