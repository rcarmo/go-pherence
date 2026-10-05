package pockettts

import (
	"math"
	"testing"
)

func TestTrainingDistillObjectiveSeedScale(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, target, noise := []float32{.2, -.3, .5}, []float32{.4, .1}, []float32{.1, -.2}
	for _, scale := range []float32{0, .25, float32(.25) * (float32(1) / 13), 1} {
		s, tm := float32(.3), float32(.8)
		xs := make([]float32, 2)
		for i := range xs {
			xs[i] = s*target[i] + (1-s)*noise[i]
		}
		v, j, e := m.ForwardTimeJVP(c, []float32{s, tm}, xs, 1)
		if e != nil {
			t.Fatal(e)
		}
		xt, desired := make([]float32, 2), make([]float32, 2)
		for i := range xt {
			xt[i] = xs[i] + (tm-s)*v[i]
			desired[i] = v[i] + (tm-s)*j[i]
		}
		_, pred, e := m.forwardTraining(c, []float32{tm, tm}, xt)
		if e != nil {
			t.Fatal(e)
		}
		in := DistillTrainingInputs{c, xs, xt, pred, desired, v, j, target, noise, s, tm, 0}
		p, e := m.buildScaledDistillTrainingTape(in, 64<<20, false, scale)
		if e != nil {
			t.Fatal(e)
		}
		got := interpretFlowTape(t, p.Tape)
		g, discard := newFlowHeadGradients(m), newFlowHeadGradients(m)
		_, ref, e := m.lsdDistillForwardBackwardInto(c, s, tm, noise, target, 0, g, discard, scale)
		if e != nil {
			t.Fatal(e)
		}
		assertSliceClose(t, "scaled condition", got[0], ref.Condition, 2e-5)
		assertSliceClose(t, "scaled target", got[1], ref.Target, 2e-5)
		named := map[string][]float32{}
		addFlowHeadGradientParameters(named, ref.Flow)
		for name, b := range p.ResidentBindings {
			for index, id := range p.Tape.Outputs {
				if id == b.Gradient {
					assertSliceClose(t, "scaled parameter "+name, got[index], named[name], 2e-5)
				}
			}
		}
	}
	for _, scale := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, e := composeScaledDistillTrainingTapes(EndpointTrainingTape{}, PrimaryTrainingTape{}, 64<<20, scale); e == nil {
			t.Fatal("scale admission")
		}
	}
}

func TestTrainingStepReductionAlreadyScaledDistill(t *testing.T) {
	in := StepReductionInputs{Frames: 1, Hidden: 2, Latent: 1, Mask: []bool{true}, DiagonalCondition: []float32{.1, .2}, DistillCondition: []float32{.3, .4}, DiagonalTarget: []float32{.2}, DistillTarget: []float32{.5}, DiagonalLogvar: []float32{.6}, DistillLogvar: []float32{.7}, RowMetrics: []float32{0, 0, 0, 0}, DistillObjectiveScaled: true}
	p, e := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 64<<20)
	if e != nil {
		t.Fatal(e)
	}
	v := interpretFlowTape(t, p)
	assertSliceClose(t, "already scaled condition", v[0], []float32{.4, .6}, 1e-7)
	assertSliceClose(t, "already scaled target", v[1], []float32{.7}, 1e-7)
	assertSliceClose(t, "raw logvar still scales", v[3], []float32{.7 * .25}, 1e-7)
}
