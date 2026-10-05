package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func stepWeightSamples(in StepReductionInputs) WeightSampleInputs {
	s := WeightSampleInputs{DiagonalTime: make([]float32, in.Frames), DistillS: make([]float32, in.Frames), DistillT: make([]float32, in.Frames), DiagonalSeed: append([]float32(nil), in.DiagonalLogvar...), DistillSeed: append([]float32(nil), in.DistillLogvar...)}
	for i := range s.DiagonalTime {
		s.DiagonalTime[i] = .3
		s.DistillS[i] = .2
		s.DistillT[i] = .8
	}
	return s
}
func TestTrainingStepWeightTapeReference(t *testing.T) {
	for _, frames := range []int{1, 3, 17} {
		for _, p := range []float32{0, .25, 1} {
			m := weightTapeFixture()
			in := stepReductionFixture(frames, max(1, frames-1))
			cfg := TrainingStepConfig{PEqual: p, EOSLossWeight: .1}
			s := stepWeightSamples(in)
			rp, err := BuildStepReductionTrainingTape(in, cfg, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			rv := interpretFlowTape(t, rp)
			scaled := s
			scaled.DiagonalSeed, scaled.DistillSeed = rv[2], rv[3]
			wp, err := m.BuildSampleWeightTrainingTape(scaled, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			wv := interpretFlowTape(t, wp.Tape)
			_, _, f64 := sampleWeightOracle(m, scaled)
			plan, err := m.BuildStepWeightTrainingTape(in, cfg, s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, plan.Tape)
			for i := 0; i < 5; i++ {
				assertSliceClose(t, "reduction output", got[i], rv[i], 2e-6)
			}
			for i, name := range wp.ParameterNames {
				assertSliceClose(t, "weight gradient", got[i+5], wv[i+2], 2e-6)
				for j, x := range got[i+5] {
					y := f64[name][j]
					if math.Abs(float64(x)-y) > 2e-6+2e-6*math.Abs(y) {
						t.Fatal("F64 weighting", name, j)
					}
				}
			}
			if err := train.ValidateResidentTapeBindings(plan.Tape, plan.ResidentBindings, 64<<20); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingStepWeightTapeAdmissionOwnership(t *testing.T) {
	m := weightTapeFixture()
	in := stepReductionFixture(3, 2)
	s := stepWeightSamples(in)
	cfg := DefaultTrainingStepConfig()
	p, err := m.BuildStepWeightTrainingTape(in, cfg, s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, x := range p.Tape.Tensors {
		n, _ := x.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.BuildStepWeightTrainingTape(in, cfg, s, bytes-1); err == nil {
		t.Fatal("combined ceiling")
	}
	if _, err := m.BuildStepWeightTrainingTape(in, cfg, s, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"frames", "sample", "mismatch", "reduction", "model", "budget"} {
		t.Run(kind, func(t *testing.T) {
			m := weightTapeFixture()
			in := stepReductionFixture(3, 2)
			s := stepWeightSamples(in)
			budget := int64(64 << 20)
			switch kind {
			case "frames":
				in.Frames = 513
			case "sample":
				s.DistillT = nil
			case "mismatch":
				s.DiagonalSeed[0]++
			case "reduction":
				in.Mask = []bool{true, false, true}
			case "model":
				m = nil
			case "budget":
				budget = 0
			}
			if _, err := m.BuildStepWeightTrainingTape(in, cfg, s, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, x := range [][]float32{in.DiagonalLogvar, in.DistillLogvar, in.RowMetrics, s.DiagonalSeed, s.DistillSeed, s.DiagonalTime, s.DistillS, s.DistillT, m.Layers[0].Weight} {
		x[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	m = weightTapeFixture()
	in = stepReductionFixture(3, 2)
	s = stepWeightSamples(in)
	priv, err := m.buildStepWeightTrainingTape(in, cfg, s, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if !f32SlicesOverlap(priv.Tape.Tensors[priv.ResidentBindings["flow.w_s_t.0.weight"].Parameter].Values, m.Layers[0].Weight) {
		t.Fatal("parameter borrow")
	}
	for _, x := range priv.Tape.Tensors {
		for _, source := range [][]float32{in.DiagonalLogvar, in.DistillLogvar, s.DiagonalSeed, s.DistillSeed, s.DiagonalTime, s.DistillS, s.DistillT} {
			if f32SlicesOverlap(x.Values, source) {
				t.Fatal("borrowed caller")
			}
		}
	}
	priv.Tape.Tensors = append(priv.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if err := train.ValidateResidentTapeBindings(priv.Tape, priv.ResidentBindings, 64<<20); err == nil {
		t.Fatal("dead source")
	}
}
func TestTrainingStepWeightBoundaryGuards(t *testing.T) {
	in := stepReductionFixture(3, 2)
	s := stepWeightSamples(in)
	p, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	fresh := func() [][]float32 { return interpretFlowTape(t, p) }
	if _, err := decodeStepWeightBoundary(fresh(), in, s); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "stats", "shape", "nan", "padding", "borrowed"} {
		v := fresh()
		switch kind {
		case "count":
			v = v[:4]
		case "stats":
			v[4] = nil
		case "shape":
			v[0] = nil
		case "nan":
			v[2][0] = float32(math.NaN())
		case "padding":
			v[2][2] = 1
		case "borrowed":
			s.DiagonalTime = append([]float32(nil), v[2]...)
			v[2] = s.DiagonalTime
		}
		if _, err := decodeStepWeightBoundary(v, in, s); err == nil {
			t.Fatal(kind)
		}
	}
}
