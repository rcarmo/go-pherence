package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// StepReductionInputs holds row boundaries after flow backward. Diagonal
// condition/target derivatives are already objective-scaled; distillation
// derivatives and both logvar branches are unscaled. Padding must be zero.
type StepReductionInputs struct {
	Frames, Hidden, Latent              int
	Mask                                []bool
	DiagonalCondition, DistillCondition []float32
	DiagonalTarget, DistillTarget       []float32
	DiagonalLogvar, DistillLogvar       []float32
	RowMetrics                          []float32 // [frames,4]: diagonal loss/square, distill loss/square
	EOS                                 float64
	// True only when primary reverse consumed objective-scaled adjoints.
	DistillObjectiveScaled bool
}
type StepReductionResult struct {
	DCondition, DTarget, DiagonalLogvar, DistillLogvar []float32
	Metrics                                            TrainingStepMetrics
}
type TrainingStepReductionBackend interface {
	EvaluateStepReduction(*FlowHeadCPU, StepReductionInputs, TrainingStepConfig) (StepReductionResult, error)
	EvaluateStepInputGradient(*FlowHeadCPU, []float32, []float32) ([]float32, error)
}

func validateStepReductionInputs(in StepReductionInputs, cfg TrainingStepConfig) (int, error) {
	if in.Frames < 1 || in.Frames > 1024 || in.Hidden < 1 || in.Hidden > 16384 || in.Latent < 1 || in.Latent > 16384 || len(in.Mask) != in.Frames || !finite64(in.EOS) || in.EOS < 0 || !isFinite(float32(in.EOS)) || !isFinite(cfg.PEqual) || cfg.PEqual < 0 || cfg.PEqual > 1 || !isFinite(cfg.EOSLossWeight) || cfg.EOSLossWeight < 0 {
		return 0, fmt.Errorf("step reduction topology/config/EOS")
	}
	arrays := [][]float32{in.DiagonalCondition, in.DistillCondition, in.DiagonalTarget, in.DistillTarget, in.DiagonalLogvar, in.DistillLogvar, in.RowMetrics}
	widths := []int{in.Hidden, in.Hidden, in.Latent, in.Latent, 1, 1, 4}
	valid := 0
	seenPadding := false
	for _, active := range in.Mask {
		if active {
			if seenPadding {
				return 0, fmt.Errorf("step reduction mask prefix")
			}
			valid++
		} else {
			seenPadding = true
		}
	}
	if valid == 0 {
		return 0, fmt.Errorf("step reduction has no active rows")
	}
	for j, x := range arrays {
		if len(x) != in.Frames*widths[j] || !finiteF32(x) {
			return 0, fmt.Errorf("step reduction boundary shape/nonfinite")
		}
		for row, active := range in.Mask {
			if !active {
				for _, v := range x[row*widths[j] : (row+1)*widths[j]] {
					if v != 0 {
						return 0, fmt.Errorf("step reduction padding must be zero")
					}
				}
			}
		}
	}
	for row := 0; row < valid; row++ {
		if in.RowMetrics[4*row+1] < 0 || in.RowMetrics[4*row+3] < 0 {
			return 0, fmt.Errorf("step reduction negative raw square")
		}
	}
	return valid, nil
}
func BuildStepReductionTrainingTape(in StepReductionInputs, cfg TrainingStepConfig, maxBytes int64) (train.Tape, error) {
	valid, err := validateStepReductionInputs(in, cfg)
	if err != nil {
		return train.Tape{}, err
	}
	if maxBytes <= 0 {
		return train.Tape{}, fmt.Errorf("step reduction ceiling required")
	}
	g := &transformerTapeGraph{constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	constant := func(x []float32) int { return g.constant([]int{len(x)}, x) }
	inv := float32(1) / float32(valid)
	diagScale, distScale := cfg.PEqual*inv, (1-cfg.PEqual)*inv
	boundaryScale := distScale
	if in.DistillObjectiveScaled {
		boundaryScale = 1
	}
	dz := g.branchAdd(constant(in.DiagonalCondition), g.branchScale(constant(in.DistillCondition), boundaryScale), 1)
	target := g.branchAdd(constant(in.DiagonalTarget), g.branchScale(constant(in.DistillTarget), boundaryScale), 1)
	dd, dt := g.branchScale(constant(in.DiagonalLogvar), diagScale), g.branchScale(constant(in.DistillLogvar), distScale)
	rowMetrics := constant(in.RowMetrics)
	ones := g.constant([]int{in.Frames}, onesTraining(in.Frames))
	sums := make([]int, 4)
	for slot := range sums {
		column := g.row(rowMetrics, g.zero(1), in.Frames, 1, 4, slot, 0)
		sums[slot] = g.branchScale(g.branchDot(column, ones), inv)
	}
	flowLoss := g.branchAdd(g.branchScale(sums[0], cfg.PEqual), g.branchScale(sums[2], 1-cfg.PEqual), 1)
	eos := g.constant([]int{1}, []float32{float32(in.EOS)})
	loss := g.branchAdd(flowLoss, g.branchScale(eos, cfg.EOSLossWeight), 1)
	metrics := g.zero(5)
	for slot, id := range []int{sums[1], sums[3], flowLoss, eos, loss} {
		metrics = g.branchAdd(metrics, g.pad(id, slot, 5), 1)
	}
	g.tape.Outputs = []int{dz, target, dd, dt, metrics}
	result, err := materializeSequenceTape(g, nil, maxBytes)
	return result.Tape, err
}
func validateStepReductionResult(r StepReductionResult, in StepReductionInputs) error {
	arrays := [][]float32{r.DCondition, r.DTarget, r.DiagonalLogvar, r.DistillLogvar}
	sizes := []int{in.Frames * in.Hidden, in.Frames * in.Latent, in.Frames, in.Frames}
	inputs := [][]float32{in.DiagonalCondition, in.DistillCondition, in.DiagonalTarget, in.DistillTarget, in.DiagonalLogvar, in.DistillLogvar, in.RowMetrics}
	for i, x := range arrays {
		if len(x) != sizes[i] || !finiteF32(x) {
			return fmt.Errorf("step reduction output shape/nonfinite")
		}
		for _, y := range arrays[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("step reduction outputs overlap")
			}
		}
		for _, y := range inputs {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("step reduction borrowed input")
			}
		}
	}
	for _, v := range []float64{r.Metrics.FlowDiagonal, r.Metrics.FlowDistill, r.Metrics.FlowLoss, r.Metrics.EOS, r.Metrics.Loss} {
		if !finite64(v) {
			return fmt.Errorf("step reduction nonfinite metrics")
		}
	}
	if r.Metrics.FlowDiagonal < 0 || r.Metrics.FlowDistill < 0 || r.Metrics.EOS < 0 {
		return fmt.Errorf("step reduction negative square/EOS")
	}
	for row, active := range in.Mask {
		if !active {
			for i, x := range arrays {
				width := 1
				if i == 0 {
					width = in.Hidden
				} else if i == 1 {
					width = in.Latent
				}
				for _, v := range x[row*width : (row+1)*width] {
					if v != 0 {
						return fmt.Errorf("step reduction inactive output")
					}
				}
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) ownsStepReduction() bool {
	return b != nil && (b.Scope == "ptx-resident-step-hybrid-v1" || b.Scope == "vulkan-resident-step-hybrid-v1" || b.ownsPruned())
}
func (b *ResidentTrainingBackend) stepReductionOwnerMatches(t *FullTrainer) bool {
	if b.ownsStepReduction() {
		return t.Flow.StepReductionBackend == b
	}
	return t.Flow.StepReductionBackend == nil
}
func (b *ResidentTrainingBackend) EvaluateStepReduction(m *FlowHeadCPU, in StepReductionInputs, cfg TrainingStepConfig) (StepReductionResult, error) {
	var r StepReductionResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.ownsStepReduction() || m != b.trainer.Flow {
		return r, b.fail(fmt.Errorf("resident step reduction owner/scope changed"))
	}
	tape, err := BuildStepReductionTrainingTape(in, cfg, b.MaxTapeBytes)
	if err != nil {
		return r, b.fail(err)
	}
	values, err := b.runUtilityTape(tape, tape.Outputs)
	if err != nil {
		return r, b.fail(err)
	}
	r = StepReductionResult{DCondition: values[0], DTarget: values[1], DiagonalLogvar: values[2], DistillLogvar: values[3], Metrics: TrainingStepMetrics{FlowDiagonal: float64(values[4][0]), FlowDistill: float64(values[4][1]), FlowLoss: float64(values[4][2]), EOS: float64(values[4][3]), Loss: float64(values[4][4])}}
	if err := validateStepReductionResult(r, in); err != nil {
		return StepReductionResult{}, b.fail(err)
	}
	return r, nil
}
func (b *ResidentTrainingBackend) EvaluateStepInputGradient(m *FlowHeadCPU, conditioner, direct []float32) ([]float32, error) {
	if err := b.admitGraph(); err != nil {
		return nil, err
	}
	if !b.ownsStepReduction() || m != b.trainer.Flow {
		return nil, b.fail(fmt.Errorf("resident step input owner/scope changed"))
	}
	if len(conditioner) < 1 || len(conditioner) > 1<<24 || len(direct) != len(conditioner) || !finiteF32(conditioner) || !finiteF32(direct) {
		return nil, b.fail(fmt.Errorf("step input gradient shape/nonfinite"))
	}
	n := len(direct)
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{n}, Values: conditioner}, {Shape: []int{n}, Values: direct}, {Shape: []int{n}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	values, err := b.runUtilityTape(tape, tape.Outputs)
	if err != nil {
		return nil, b.fail(err)
	}
	if f32SlicesOverlap(values[0], conditioner) || f32SlicesOverlap(values[0], direct) {
		return nil, b.fail(fmt.Errorf("step input borrowed gradient"))
	}
	return values[0], nil
}
