package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type DiagonalForwardInputs struct {
	Condition, Target, Noise []float32
	Time                     float32
}
type DiagonalForwardResult struct{ Input, Desired, Prediction []float32 }
type DiagonalForwardTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}
type TrainingDiagonalForwardBackend interface {
	DiagonalForwardEnabled() bool
	EvaluateDiagonalForward(*FlowHeadCPU, DiagonalForwardInputs) (DiagonalForwardResult, error)
}

func (b *ResidentTrainingBackend) DiagonalForwardEnabled() bool {
	return b != nil && (b.weightRootsEnabled() || b.Scope == "ptx-resident-diagforward-hybrid-v1" || b.Scope == "vulkan-resident-diagforward-hybrid-v1")
}
func (m *FlowHeadCPU) BuildDiagonalForwardTape(in DiagonalForwardInputs, maxBytes int64) (DiagonalForwardTape, error) {
	return m.buildDiagonalForwardTape(in, maxBytes, false)
}
func (m *FlowHeadCPU) buildDiagonalForwardTape(in DiagonalForwardInputs, maxBytes int64, borrow bool) (DiagonalForwardTape, error) {
	var result DiagonalForwardTape
	if m == nil || len(m.Time) != 2 {
		return result, fmt.Errorf("diagonal forward topology")
	}
	interpolation, err := BuildBranchTrainingTape("interpolate", [][]float32{in.Target, in.Noise}, in.Time, maxBytes)
	if err != nil {
		return result, err
	}
	zero := make([]float32, len(in.Target))
	flow, err := m.buildFlowDualTrainingTape(in.Condition, []float32{in.Time, in.Time}, zero, -1, zero, zero, maxBytes, true, borrow)
	if err != nil {
		return result, err
	}
	if err := train.ValidateResidentTapeBindings(flow.Tape, flow.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	t := interpolation
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	outputs, bindings, err := appendAdmittedFlowGraph(&t, flow, interpolation.Outputs[0], nil)
	if err != nil {
		return result, err
	}
	t.Outputs = []int{interpolation.Outputs[0], interpolation.Outputs[1], outputs[0]}
	for _, name := range flow.ParameterNames {
		t.Outputs = append(t.Outputs, bindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return result, err
	}
	return DiagonalForwardTape{t, bindings}, nil
}
func validateDiagonalForwardResult(r DiagonalForwardResult, in DiagonalForwardInputs) error {
	values := [][]float32{r.Input, r.Desired, r.Prediction}
	for i, x := range values {
		if len(x) != len(in.Target) || !finiteF32(x) {
			return fmt.Errorf("diagonal forward result shape/nonfinite")
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("diagonal forward aliased outputs")
			}
		}
		for _, source := range [][]float32{in.Condition, in.Target, in.Noise} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("diagonal forward borrowed output")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateDiagonalForward(m *FlowHeadCPU, in DiagonalForwardInputs) (DiagonalForwardResult, error) {
	var r DiagonalForwardResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.DiagonalForwardEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("diagonal forward owner/scope"))
	}
	p, err := m.buildDiagonalForwardTape(in, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], false, 1)
	if err != nil {
		return r, b.fail(err)
	}
	if len(out) != 3 {
		return r, b.fail(fmt.Errorf("diagonal forward boundary count"))
	}
	r = DiagonalForwardResult{out[0], out[1], out[2]}
	if err := validateDiagonalForwardResult(r, in); err != nil {
		return DiagonalForwardResult{}, b.fail(err)
	}
	return r, nil
}
