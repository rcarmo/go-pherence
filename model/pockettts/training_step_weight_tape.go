package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private step capability: return ordinary reduction boundaries while weighting
// consumes the scaled logvar seed IDs directly and accumulates all roots once.
type trainingStepWeightBackend interface {
	stepWeightEnabled() bool
	evaluateStepWeight(*FlowHeadCPU, *LSDWeightMLP, StepReductionInputs, TrainingStepConfig, WeightSampleInputs) (StepReductionResult, error)
}
type StepWeightTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}

func (b *ResidentTrainingBackend) stepWeightEnabled() bool {
	return b != nil && (b.stepBackboneEnabled() || b.Scope == "ptx-resident-stepweight-hybrid-v1" || b.Scope == "vulkan-resident-stepweight-hybrid-v1")
}
func (m *LSDWeightMLP) BuildStepWeightTrainingTape(in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs, maxBytes int64) (StepWeightTrainingTape, error) {
	return m.buildStepWeightTrainingTape(in, cfg, samples, maxBytes, false)
}
func (m *LSDWeightMLP) buildStepWeightTrainingTape(in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs, maxBytes int64, borrow bool) (StepWeightTrainingTape, error) {
	var result StepWeightTrainingTape
	if in.Frames < 1 || in.Frames > 512 {
		return result, fmt.Errorf("step weighting frame ceiling")
	}
	if err := samples.validate(2 * in.Frames); err != nil {
		return result, err
	}
	// Caller seeds cannot silently disagree with the reduction or hide NaNs. The
	// graph scales the admitted raw branches exactly once, before MLP seed packing.
	for i := 0; i < in.Frames; i++ {
		if len(in.DiagonalLogvar) != in.Frames || len(in.DistillLogvar) != in.Frames || samples.DiagonalSeed[i] != in.DiagonalLogvar[i] || samples.DistillSeed[i] != in.DistillLogvar[i] {
			return result, fmt.Errorf("step weighting raw seed mismatch")
		}
	}
	reduction, err := BuildStepReductionTrainingTape(in, cfg, maxBytes)
	if err != nil {
		return result, err
	}
	weight, err := m.buildWeightTrainingTape(nil, nil, 2*in.Frames, maxBytes, &samples, borrow)
	if err != nil {
		return result, err
	}
	if err := train.ValidateResidentTapeBindings(weight.Tape, weight.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	t := reduction
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	offset := len(t.Tensors)
	t.Tensors = append(t.Tensors, weight.Tape.Tensors...)
	mapping := make([]int, len(weight.Tape.Tensors))
	for i := range mapping {
		mapping[i] = offset + i
	}
	for i, id := range weight.sampleSeedIDs {
		source := reduction.Outputs[2+i]
		a, _ := weight.Tape.Tensors[id].Elements()
		z, _ := t.Tensors[source].Elements()
		if a != z {
			return result, fmt.Errorf("step weighting seed edge")
		}
		mapping[id] = source
	}
	for _, c := range weight.Tape.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		for i, id := range c.Inputs {
			c.Inputs[i] = mapping[id]
		}
		c.Output = mapping[c.Output]
		t.Commands = append(t.Commands, c)
	}
	t.Outputs = append([]int(nil), reduction.Outputs...)
	bindings := map[string]train.ResidentTapeBinding{}
	for _, name := range weight.ParameterNames {
		b := weight.ResidentBindings[name]
		binding := train.ResidentTapeBinding{Parameter: mapping[b.Parameter], Gradient: mapping[b.Gradient]}
		bindings[name] = binding
		t.Outputs = append(t.Outputs, binding.Gradient)
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return result, err
	}
	return StepWeightTrainingTape{t, bindings}, nil
}
func decodeStepWeightBoundary(out [][]float32, in StepReductionInputs, samples WeightSampleInputs) (StepReductionResult, error) {
	if len(out) != 5 || len(out[4]) != 5 {
		return StepReductionResult{}, fmt.Errorf("step weighting boundary count")
	}
	r := StepReductionResult{out[0], out[1], out[2], out[3], TrainingStepMetrics{FlowDiagonal: float64(out[4][0]), FlowDistill: float64(out[4][1]), FlowLoss: float64(out[4][2]), EOS: float64(out[4][3]), Loss: float64(out[4][4])}}
	if err := validateStepReductionResult(r, in); err != nil {
		return StepReductionResult{}, err
	}
	for _, x := range [][]float32{r.DCondition, r.DTarget, r.DiagonalLogvar, r.DistillLogvar} {
		for _, source := range [][]float32{samples.DiagonalTime, samples.DistillS, samples.DistillT, samples.DiagonalSeed, samples.DistillSeed} {
			if f32SlicesOverlap(x, source) {
				return StepReductionResult{}, fmt.Errorf("step weighting borrowed sample")
			}
		}
	}
	return r, nil
}
func (b *ResidentTrainingBackend) evaluateStepWeight(flow *FlowHeadCPU, weight *LSDWeightMLP, in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs) (StepReductionResult, error) {
	var r StepReductionResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.stepWeightEnabled() || b.trainer.Flow != flow || b.trainer.Weighting != weight || flow.StepReductionBackend != b || weight.TrainingBackend != b {
		return r, b.fail(fmt.Errorf("step weighting owner/scope"))
	}
	p, err := weight.buildStepWeightTrainingTape(in, cfg, samples, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:5], true, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodeStepWeightBoundary(out, in, samples)
	if err != nil {
		return StepReductionResult{}, b.fail(err)
	}
	return r, nil
}
