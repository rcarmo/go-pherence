package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private step hook carries known log variance into a read-only forward guard.
// Public forward interfaces/results and legacy scopes keep their old contract.
type trainingDiagonalAdmissionBackend interface {
	diagonalAdmissionEnabled() bool
	evaluateDiagonalAdmitted(*FlowHeadCPU, DiagonalForwardInputs, float32) (DiagonalForwardResult, error)
}

func (b *ResidentTrainingBackend) diagonalAdmissionEnabled() bool {
	return b != nil && (b.distillAdmissionEnabled() || b.Scope == "ptx-resident-diagadmission-hybrid-v1" || b.Scope == "vulkan-resident-diagadmission-hybrid-v1")
}
func appendDiagonalAdmission(p DiagonalForwardTape, logvar float32, maxBytes int64) (DiagonalForwardTape, error) {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return DiagonalForwardTape{}, err
	}
	if len(p.Tape.Outputs) < 3 {
		return DiagonalForwardTape{}, fmt.Errorf("diagonal admission source outputs")
	}
	prediction, desired := p.Tape.Outputs[2], p.Tape.Outputs[1]
	n, err := p.Tape.Tensors[prediction].Elements()
	if err != nil {
		return DiagonalForwardTape{}, err
	}
	cutoff, err := train.LSDPredictionCutoff(n, logvar)
	if err != nil {
		return DiagonalForwardTape{}, err
	}
	// Match the original loss admission's standalone logical ceiling as well as
	// the complete expanded forward graph ceiling; graph budgets are not guards.
	if maxBytes <= 0 || int64(4*(3*n+4)) > maxBytes {
		return DiagonalForwardTape{}, fmt.Errorf("LSD loss tape byte ceiling")
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
		return DiagonalForwardTape{}, err
	}
	return DiagonalForwardTape{Tape: t, ResidentBindings: p.ResidentBindings}, nil
}
func validateDiagonalAdmissionFlag(values [][]float32) error {
	if len(values) != 4 || len(values[3]) != 1 || !isFinite(values[3][0]) || values[3][0] != 0 {
		return fmt.Errorf("diagonal prediction admission failed")
	}
	for _, x := range values[:3] {
		if f32SlicesOverlap(values[3], x) {
			return fmt.Errorf("diagonal admission aliased flag")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) evaluateDiagonalAdmitted(m *FlowHeadCPU, in DiagonalForwardInputs, logvar float32) (DiagonalForwardResult, error) {
	var result DiagonalForwardResult
	if err := b.admitGraph(); err != nil {
		return result, err
	}
	if !b.diagonalAdmissionEnabled() || m != b.trainer.Flow || m.TrainingBackend != b || m.BranchBackend != b {
		return result, b.fail(fmt.Errorf("diagonal admission owner/scope"))
	}
	if b.deviceHandoffEnabled() && b.handoffLease != nil {
		return b.evaluateDeviceDiagonalForward(m, in, logvar)
	}
	p, err := m.buildDiagonalForwardTape(in, b.MaxTapeBytes, true)
	if err != nil {
		return result, b.fail(err)
	}
	p, err = appendDiagonalAdmission(p, logvar, b.MaxTapeBytes)
	if err != nil {
		return result, b.fail(err)
	}
	outputs := append([]int(nil), p.Tape.Outputs[:3]...)
	outputs = append(outputs, p.Tape.Outputs[len(p.Tape.Outputs)-1])
	var values [][]float32
	if b.predictionHandoffEnabled() && b.handoffLease != nil {
		values, err = b.beginPredictionHandoff(p.Tape, p.ResidentBindings, outputs, in.Condition, "diagonal", 3)
	} else {
		values, err = b.runFlowHandoffTape(p.Tape, p.ResidentBindings, outputs, in.Condition)
	}
	if err != nil {
		return result, b.fail(err)
	}
	if err := validateDiagonalAdmissionFlag(values); err != nil {
		return result, b.fail(err)
	}
	for _, source := range [][]float32{in.Condition, in.Target, in.Noise} {
		if f32SlicesOverlap(values[3], source) {
			return result, b.fail(fmt.Errorf("diagonal admission borrowed flag"))
		}
	}
	result = DiagonalForwardResult{Input: values[0], Desired: values[1], Prediction: values[2]}
	if err := validateDiagonalForwardResult(result, in); err != nil {
		return DiagonalForwardResult{}, b.fail(err)
	}
	// Redundant host check retains the old contract until native cutoff/FTZ
	// qualification and the prediction-device handoff are separately approved.
	if err := admitLSDLossTraining(result.Prediction, result.Desired, logvar, b.MaxTapeBytes); err != nil {
		return DiagonalForwardResult{}, b.fail(err)
	}
	return result, nil
}
