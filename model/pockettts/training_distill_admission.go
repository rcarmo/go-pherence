package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private step-only hook. Public minimal-LSD and forward calls retain their
// original outputs/admission path and receive no new scalar diagnostic.
type trainingDistillAdmissionBackend interface {
	distillAdmissionEnabled() bool
	evaluateDistillAdmitted(*FlowHeadCPU, DistillForwardInputs, float32) (DistillForwardResult, error)
}

func (b *ResidentTrainingBackend) distillAdmissionEnabled() bool {
	return b != nil && (b.flowHandoffEnabled() || b.Scope == "ptx-resident-distilladmission-hybrid-v1" || b.Scope == "vulkan-resident-distilladmission-hybrid-v1")
}
func appendDistillAdmission(p DistillForwardTape, logvar float32, maxBytes int64) (DistillForwardTape, error) {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return DistillForwardTape{}, err
	}
	if len(p.Tape.Outputs) < 6 {
		return DistillForwardTape{}, fmt.Errorf("distill admission source outputs")
	}
	prediction, desired := p.Tape.Outputs[5], p.Tape.Outputs[4]
	n, err := p.Tape.Tensors[prediction].Elements()
	if err != nil {
		return DistillForwardTape{}, err
	}
	cutoff, err := train.LSDPredictionCutoff(n, logvar)
	if err != nil {
		return DistillForwardTape{}, err
	}
	if maxBytes <= 0 || int64(4*(3*n+4)) > maxBytes {
		return DistillForwardTape{}, fmt.Errorf("LSD loss tape byte ceiling")
	}
	t := p.Tape
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	t.Outputs = append([]int(nil), t.Outputs...)
	constant := len(t.Tensors)
	flag := constant + 1
	t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{cutoff}}, train.TensorSpec{Shape: []int{1}})
	t.Commands = append(t.Commands, train.Command{Op: "lsd-loss", Inputs: []int{prediction, desired, constant}, Output: flag, Mode: 2, Scale: logvar})
	t.Outputs = append(t.Outputs, flag)
	if err := train.ValidateResidentTapeBindings(t, p.ResidentBindings, maxBytes); err != nil {
		return DistillForwardTape{}, err
	}
	return DistillForwardTape{Tape: t, ResidentBindings: p.ResidentBindings}, nil
}
func validateDistillAdmissionFlag(values [][]float32, in DistillForwardInputs) error {
	if len(values) != 7 || len(values[6]) != 1 || !isFinite(values[6][0]) || values[6][0] != 0 {
		return fmt.Errorf("distill prediction admission failed")
	}
	for _, x := range values[:6] {
		if f32SlicesOverlap(values[6], x) {
			return fmt.Errorf("distill admission aliased flag")
		}
	}
	for _, x := range [][]float32{in.Condition, in.Target, in.Noise} {
		if f32SlicesOverlap(values[6], x) {
			return fmt.Errorf("distill admission borrowed flag")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) evaluateDistillAdmitted(m *FlowHeadCPU, in DistillForwardInputs, logvar float32) (DistillForwardResult, error) {
	var result DistillForwardResult
	if err := b.admitGraph(); err != nil {
		return result, err
	}
	if !b.distillAdmissionEnabled() || m != b.trainer.Flow || m.TrainingBackend != b || m.BranchBackend != b {
		return result, b.fail(fmt.Errorf("distill admission owner/scope"))
	}
	if b.deviceHandoffEnabled() && b.handoffLease != nil {
		return b.evaluateDeviceDistillForward(m, in, logvar)
	}
	p, err := m.buildDistillForwardTape(in, b.MaxTapeBytes, true)
	if err != nil {
		return result, b.fail(err)
	}
	p, err = appendDistillAdmission(p, logvar, b.MaxTapeBytes)
	if err != nil {
		return result, b.fail(err)
	}
	outputs := append([]int(nil), p.Tape.Outputs[:6]...)
	outputs = append(outputs, p.Tape.Outputs[len(p.Tape.Outputs)-1])
	var values [][]float32
	if b.predictionHandoffEnabled() && b.handoffLease != nil {
		values, err = b.beginPredictionHandoff(p.Tape, p.ResidentBindings, outputs, in.Condition, "distill", 6)
	} else {
		values, err = b.runFlowHandoffTape(p.Tape, p.ResidentBindings, outputs, in.Condition)
	}
	if err != nil {
		return result, b.fail(err)
	}
	if err := validateDistillAdmissionFlag(values, in); err != nil {
		return result, b.fail(err)
	}
	result, err = decodeDistillForwardBoundary(values[:6], in)
	if err != nil {
		return DistillForwardResult{}, b.fail(err)
	}
	// Preserve the old host guard pending native cutoff/FTZ and handoff approval.
	if err := admitLSDLossTraining(result.Prediction, result.Desired, logvar, b.MaxTapeBytes); err != nil {
		return DistillForwardResult{}, b.fail(err)
	}
	return result, nil
}
