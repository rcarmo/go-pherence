package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingEndpointTapeComposition(t *testing.T) {
	for _, time := range []float32{0, .4, 1} {
		for _, delta := range []float32{0, time} {
			for _, lv := range []float32{-2, 0, 2} {
				m := tinyTrainableFlowHead()
				condition := []float32{.2, -.3, .5}
				input := []float32{.3, -.2}
				desired, v, tan := []float32{.4, -.1}, []float32{.2, .5}, []float32{-.1, .3}
				forward, err := m.BuildFlowDeviceTimeTrainingTape(condition, []float32{time, time}, input, -1, make([]float32, 2), make([]float32, 2), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				fv := interpretFlowTape(t, forward.Tape)
				loss, err := BuildLSDLossTrainingTape(desired, fv[0], lv, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				lossV := interpretFlowTape(t, loss)
				negative, err := BuildBranchTrainingTape("negate", [][]float32{lossV[1]}, 0, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				neg := interpretFlowTape(t, negative)
				backward, err := m.BuildFlowDeviceTimeTrainingTape(condition, []float32{time, time}, input, -1, neg[0], make([]float32, 2), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				bv := interpretFlowTape(t, backward.Tape)
				mixed, err := BuildBranchTrainingTape("mixed", [][]float32{lossV[1], bv[3], v, tan}, delta, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				mv := interpretFlowTape(t, mixed)
				p, err := m.BuildEndpointTrainingTape(condition, time, input, fv[0], desired, v, tan, lv, delta, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, p.Tape)
				for i, x := range [][]float32{bv[3], mv[0], mv[1], lossV[0], mv[2]} {
					assertSliceClose(t, "composed endpoint", got[i], x, 2e-5)
				}
				if len(p.ResidentBindings) != len(backward.ResidentBindings) {
					t.Fatal("full inventory")
				}
				if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
func TestTrainingEndpointTapeIndependentCPU(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, x := []float32{.2, -.3, .5}, []float32{.3, -.2}
	desired, v, tan := []float32{.4, -.1}, []float32{.2, .5}, []float32{-.1, .3}
	time, delta, lv := float32(.4), float32(.3), float32(.2)
	tape, pred, err := m.forwardTraining(c, []float32{time, time}, x)
	if err != nil {
		t.Fatal(err)
	}
	dr, de := make([]float32, 2), make([]float32, 2)
	square := float64(0)
	for i, p := range pred {
		d := float64(desired[i]) - float64(p)
		square += d * d
		dr[i] = float32(2 * d * math.Exp(float64(lv)) / 2)
		de[i] = -dr[i]
	}
	g := newFlowHeadGradients(m)
	_, _, dx, err := m.backwardTraining(tape, de, g)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildEndpointTrainingTape(c, time, x, pred, desired, v, tan, lv, delta, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	out := interpretFlowTape(t, p.Tape)
	assertSliceClose(t, "CPU endpoint input", out[0], dx, 2e-5)
	gv, gt := make([]float32, 2), make([]float32, 2)
	dd := float64(0)
	for i := range dr {
		gv[i] = dr[i] + delta*dx[i]
		gt[i] = delta * dr[i]
		dd += float64(dr[i])*float64(tan[i]) + float64(dx[i])*float64(v[i])
	}
	assertSliceClose(t, "CPU primary gv", out[1], gv, 2e-5)
	assertSliceClose(t, "CPU primary gt", out[2], gt, 2e-5)
	wants := []float64{square, math.Exp(float64(lv))*square/2 - float64(lv), math.Exp(float64(lv))*square/2 - 1}
	for i, w := range wants {
		if math.Abs(float64(out[3][i])-w) > 2e-6+2e-6*math.Abs(w) {
			t.Fatal("F64 stats", i)
		}
	}
	if math.Abs(float64(out[4][0])-dd) > 2e-6+2e-6*math.Abs(dd) {
		t.Fatal("F64 delta")
	}
}
func TestTrainingEndpointTapeAdmissionOwnership(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, x, pred, desired, v, tan := []float32{.2, -.3, .5}, []float32{.3, -.2}, []float32{.2, -.1}, []float32{.4, -.1}, []float32{.2, .5}, []float32{-.1, .3}
	p, err := m.BuildEndpointTrainingTape(c, .4, x, pred, desired, v, tan, 0, .3, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildEndpointTrainingTape(c, .4, x, pred, desired, v, tan, 0, .3, bytes-1); err == nil {
		t.Fatal("full bytes")
	}
	if _, err := m.BuildEndpointTrainingTape(c, .4, x, pred, desired, v, tan, 0, .3, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"time", "delta", "velocity", "tangent", "prediction", "logvar", "condition", "shape", "budget"} {
		t.Run(kind, func(t *testing.T) {
			cc := append([]float32(nil), c...)
			pr := append([]float32(nil), pred...)
			vv, tt := append([]float32(nil), v...), append([]float32(nil), tan...)
			time, delta, lv := float32(.4), float32(.3), float32(0)
			budget := int64(64 << 20)
			switch kind {
			case "time":
				time = -1
			case "delta":
				delta = .5
			case "velocity":
				vv = nil
			case "tangent":
				tt[0] = float32(math.NaN())
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
			if _, err := m.BuildEndpointTrainingTape(cc, time, x, pr, desired, vv, tt, lv, delta, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, a := range [][]float32{c, x, pred, desired, v, tan, m.Input.Weight} {
		a[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, a := range before {
		assertSliceClose(t, "public owned", after[i], a, 0)
	}
}
func TestTrainingEndpointResultGuards(t *testing.T) {
	c, x := make([]float32, 3), make([]float32, 2)
	fresh := func() EndpointTrainingResult {
		return EndpointTrainingResult{make([]float32, 2), make([]float32, 2), make([]float32, 2), 1, 0, 0, 0}
	}
	if err := validateEndpointResult(fresh(), c, x, x, x, x, x); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "square", "overlap", "source"} {
		r := fresh()
		switch kind {
		case "shape":
			r.GV = nil
		case "nan":
			r.DDelta = float32(math.NaN())
		case "square":
			r.RawSquare = -1
		case "overlap":
			r.GT = r.GV
		case "source":
			r.DInput = x
		}
		if err := validateEndpointResult(r, c, x, x, x, x, x); err == nil {
			t.Fatal(kind)
		}
	}
}
