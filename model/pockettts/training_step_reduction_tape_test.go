package pockettts

import (
	"math"
	"testing"
)

func stepReductionFixture(frames, valid int) StepReductionInputs {
	in := StepReductionInputs{Frames: frames, Hidden: 2, Latent: 1, Mask: make([]bool, frames), EOS: .25, DiagonalCondition: make([]float32, frames*2), DistillCondition: make([]float32, frames*2), DiagonalTarget: make([]float32, frames), DistillTarget: make([]float32, frames), DiagonalLogvar: make([]float32, frames), DistillLogvar: make([]float32, frames), RowMetrics: make([]float32, frames*4)}
	for i := 0; i < valid; i++ {
		in.Mask[i] = true
		in.DiagonalCondition[i*2], in.DiagonalCondition[i*2+1] = 1, 2
		in.DistillCondition[i*2], in.DistillCondition[i*2+1] = 4, -2
		in.DiagonalTarget[i] = 3
		in.DistillTarget[i] = -4
		in.DiagonalLogvar[i] = 2
		in.DistillLogvar[i] = -2
		copy(in.RowMetrics[i*4:], []float32{2, 4, 6, 8})
	}
	return in
}
func decodeStepReductionTest(values [][]float32) StepReductionResult {
	return StepReductionResult{DCondition: values[0], DTarget: values[1], DiagonalLogvar: values[2], DistillLogvar: values[3], Metrics: TrainingStepMetrics{FlowDiagonal: float64(values[4][0]), FlowDistill: float64(values[4][1]), FlowLoss: float64(values[4][2]), EOS: float64(values[4][3]), Loss: float64(values[4][4])}}
}
func TestTrainingStepReductionTapeReference(t *testing.T) {
	in := stepReductionFixture(3, 2)
	cfg := TrainingStepConfig{PEqual: .75, EOSLossWeight: .5}
	tape, err := BuildStepReductionTrainingTape(in, cfg, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	r := decodeStepReductionTest(interpretFlowTape(t, tape))
	if err := validateStepReductionResult(r, in); err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "reduced condition", r.DCondition, []float32{1.5, 1.75, 1.5, 1.75, 0, 0}, 0)
	assertSliceClose(t, "reduced target", r.DTarget, []float32{2.5, 2.5, 0}, 0)
	assertSliceClose(t, "diag logvar", r.DiagonalLogvar, []float32{.75, .75, 0}, 0)
	assertSliceClose(t, "distill logvar", r.DistillLogvar, []float32{-.25, -.25, 0}, 0)
	if r.Metrics != (TrainingStepMetrics{FlowDiagonal: 4, FlowDistill: 8, FlowLoss: 3, EOS: .25, Loss: 3.125}) {
		t.Fatal("independent small metrics", r.Metrics)
	}
	var maxMetric, maxGradient float64
	for _, frames := range []int{1, 17, 257, 1024} {
		for _, p := range []float32{0, .25, .75, 1} {
			in := stepReductionFixture(frames, frames)
			for i := range in.RowMetrics {
				if i%4 == 0 || i%4 == 2 {
					in.RowMetrics[i] = float32(i%23-11) * .0037
				} else {
					in.RowMetrics[i] = float32(i%19) * .031
				}
			}
			cfg := TrainingStepConfig{PEqual: p, EOSLossWeight: .1}
			tape, err := BuildStepReductionTrainingTape(in, cfg, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			r := decodeStepReductionTest(interpretFlowTape(t, tape))
			var means [4]float64
			for row := 0; row < frames; row++ {
				for j := range means {
					means[j] += float64(in.RowMetrics[row*4+j]) / float64(frames)
				}
			}
			flow := float64(p)*means[0] + (1-float64(p))*means[2]
			loss := flow + float64(cfg.EOSLossWeight)*in.EOS
			for i, got := range []float64{r.Metrics.FlowDiagonal, r.Metrics.FlowDistill, r.Metrics.FlowLoss, r.Metrics.EOS, r.Metrics.Loss} {
				want := []float64{means[1], means[3], flow, in.EOS, loss}[i]
				e := math.Abs(got - want)
				maxMetric = math.Max(maxMetric, e)
				if e > 3e-7+2e-6*math.Abs(want) {
					t.Fatal("F64 metrics", frames, p, i, got, want, e)
				}
			}
			for i, v := range r.DCondition {
				want := float64(in.DiagonalCondition[i]) + float64(in.DistillCondition[i])*(1-float64(p))/float64(frames)
				e := math.Abs(float64(v) - want)
				maxGradient = math.Max(maxGradient, e)
				if e > 2e-7 {
					t.Fatal("F64 gradient", frames, p, i, e)
				}
			}
		}
	}
	t.Logf("Step reduction F32/F64 max metric=%g gradient=%g across frames1/17/257/1024 and weights0/.25/.75/1", maxMetric, maxGradient)
}
func TestTrainingStepReductionTapeAdmission(t *testing.T) {
	for _, mutate := range []func(*StepReductionInputs){func(in *StepReductionInputs) { in.Frames = 0 }, func(in *StepReductionInputs) { in.Hidden = 16385 }, func(in *StepReductionInputs) { in.Mask = nil }, func(in *StepReductionInputs) { in.Mask = []bool{true, false, true} }, func(in *StepReductionInputs) { in.Mask = []bool{false, false, false} }, func(in *StepReductionInputs) { in.DiagonalCondition = nil }, func(in *StepReductionInputs) { in.DiagonalTarget[2] = 1 }, func(in *StepReductionInputs) { in.EOS = math.Inf(1) }, func(in *StepReductionInputs) { in.RowMetrics[0] = float32(math.NaN()) }, func(in *StepReductionInputs) { in.RowMetrics[1] = -1 }} {
		in := stepReductionFixture(3, 2)
		mutate(&in)
		if _, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 1<<20); err == nil {
			t.Fatal("bad reduction")
		}
	}
	in := stepReductionFixture(3, 2)
	for _, cfg := range []TrainingStepConfig{{PEqual: -1}, {PEqual: 2}, {PEqual: .75, EOSLossWeight: -1}} {
		if _, err := BuildStepReductionTrainingTape(in, cfg, 1<<20); err == nil {
			t.Fatal("bad config")
		}
	}
	if _, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 1); err == nil {
		t.Fatal("budget")
	}
	tape, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	r := decodeStepReductionTest(interpretFlowTape(t, tape))
	for _, mutate := range []func(*StepReductionResult){func(r *StepReductionResult) { r.DTarget = nil }, func(r *StepReductionResult) { r.DCondition = in.DiagonalCondition }, func(r *StepReductionResult) { r.DistillLogvar = r.DiagonalLogvar }, func(r *StepReductionResult) { r.Metrics.Loss = math.Inf(1) }, func(r *StepReductionResult) { r.Metrics.EOS = -1 }, func(r *StepReductionResult) { r.DTarget = append([]float32(nil), r.DTarget...); r.DTarget[2] = 1 }} {
		bad := r
		mutate(&bad)
		if validateStepReductionResult(bad, in) == nil {
			t.Fatal("bad result")
		}
	}
}
