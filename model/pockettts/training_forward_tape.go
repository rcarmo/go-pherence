package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type DistillForwardInputs struct {
	Condition, Target, Noise []float32
	S, T                     float32
}
type DistillForwardResult struct{ PrimaryInput, Velocity, Tangent, EndpointInput, Desired, Prediction []float32 }
type DistillForwardTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}
type TrainingDistillForwardBackend interface {
	DistillForwardEnabled() bool
	EvaluateDistillForward(*FlowHeadCPU, DistillForwardInputs) (DistillForwardResult, error)
}

func (b *ResidentTrainingBackend) DistillForwardEnabled() bool {
	return b != nil && (b.DiagonalForwardEnabled() || b.Scope == "ptx-resident-forwardfused-hybrid-v1" || b.Scope == "vulkan-resident-forwardfused-hybrid-v1")
}
func (m *FlowHeadCPU) BuildDistillForwardTape(in DistillForwardInputs, maxBytes int64) (DistillForwardTape, error) {
	return m.buildDistillForwardTape(in, maxBytes, false)
}
func (m *FlowHeadCPU) buildDistillForwardTape(in DistillForwardInputs, maxBytes int64, borrow bool) (DistillForwardTape, error) {
	var result DistillForwardTape
	if m == nil || len(m.Time) != 2 || !isFinite(in.S) || !isFinite(in.T) || in.S < 0 || in.T > 1 || in.S > in.T {
		return result, fmt.Errorf("distill forward time/topology")
	}
	// Preserve interpolation primitive order. Metadata/owned constants remain
	// included in the full-source ceiling, including its unused desired output.
	interpolation, err := BuildBranchTrainingTape("interpolate", [][]float32{in.Target, in.Noise}, in.S, maxBytes)
	if err != nil {
		return result, err
	}
	n := len(in.Target)
	zero := make([]float32, n)
	var sources [2]FlowDualTrainingTape
	times := [2][2]float32{{in.S, in.T}, {in.T, in.T}}
	selected := [2]int{1, -1}
	for i := range sources {
		sources[i], err = m.buildFlowDualTrainingTape(in.Condition, times[i][:], zero, selected[i], zero, zero, maxBytes, true, borrow)
		if err != nil {
			return result, err
		}
	}
	return composeDistillForwardTapes(interpolation, sources[0], sources[1], in.T-in.S, maxBytes)
}

// Full source admission precedes composition; incompatible component graphs
// fail before remapping and before pruning can hide their unused tensors.
func composeDistillForwardTapes(interpolation train.Tape, primary, endpoint FlowDualTrainingTape, delta float32, maxBytes int64) (DistillForwardTape, error) {
	var result DistillForwardTape
	if !isFinite(delta) || delta < 0 || delta > 1 {
		return result, fmt.Errorf("distill forward delta")
	}
	if err := interpolation.Validate(maxBytes); err != nil {
		return result, err
	}
	if err := train.ValidateResidentTapeBindings(primary.Tape, primary.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	if err := train.ValidateResidentTapeBindings(endpoint.Tape, endpoint.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	if len(interpolation.Outputs) < 1 || len(primary.Tape.Outputs) < 2 || len(endpoint.Tape.Outputs) < 1 {
		return result, fmt.Errorf("distill forward source outputs")
	}
	n, _ := interpolation.Tensors[interpolation.Outputs[0]].Elements()
	if len(primary.ParameterNames) != len(primary.ResidentBindings) {
		return result, fmt.Errorf("distill forward parameter names")
	}
	seen := map[string]bool{}
	for _, name := range primary.ParameterNames {
		if _, ok := primary.ResidentBindings[name]; !ok || seen[name] {
			return result, fmt.Errorf("distill forward parameter names")
		}
		seen[name] = true
	}
	tensors := len(interpolation.Tensors) + len(primary.Tape.Tensors) + len(endpoint.Tape.Tensors) + 2
	commands := len(interpolation.Commands) + len(primary.Tape.Commands) + len(endpoint.Tape.Commands) + 2
	if tensors > 4096 || commands > 4096 {
		return result, fmt.Errorf("distill forward combined graph ceiling")
	}
	t := interpolation
	t.Tensors = make([]train.TensorSpec, len(interpolation.Tensors), tensors)
	copy(t.Tensors, interpolation.Tensors)
	t.Commands = make([]train.Command, len(interpolation.Commands), commands)
	copy(t.Commands, interpolation.Commands)
	// Append already admitted graphs with explicit input/parameter remaps. Do not
	// prune until combined source and all bindings have passed admission.
	xs := interpolation.Outputs[0]
	po, bindings, err := appendAdmittedFlowGraph(&t, primary, xs, nil)
	if err != nil {
		return result, err
	}
	g := &flowTapeGraph{tape: t, params: map[string]int{}, adj: map[int]int{}}
	xt := g.command("vector", []int{xs, po[0]}, []int{n}, 0, 0, 0, delta, 0)
	desired := g.command("vector", []int{po[0], po[1]}, []int{n}, 0, 0, 0, delta, 0)
	t = g.tape
	eo, _, err := appendAdmittedFlowGraph(&t, endpoint, xt, bindings)
	if err != nil {
		return result, err
	}
	t.Outputs = make([]int, 6, 6+len(primary.ParameterNames))
	copy(t.Outputs, []int{xs, po[0], po[1], xt, desired, eo[0]})
	for _, name := range primary.ParameterNames {
		t.Outputs = append(t.Outputs, bindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return result, err
	}
	return DistillForwardTape{t, bindings}, nil
}
func validateDistillForwardResult(r DistillForwardResult, in DistillForwardInputs) error {
	values := [][]float32{r.PrimaryInput, r.Velocity, r.Tangent, r.EndpointInput, r.Desired, r.Prediction}
	for i, x := range values {
		if len(x) != len(in.Target) || !finiteF32(x) {
			return fmt.Errorf("distill forward result shape/nonfinite")
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("distill forward aliased outputs")
			}
		}
		for _, source := range [][]float32{in.Condition, in.Target, in.Noise} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("distill forward borrowed output")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateDistillForward(m *FlowHeadCPU, in DistillForwardInputs) (DistillForwardResult, error) {
	var r DistillForwardResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.DistillForwardEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("distill forward owner/scope"))
	}
	p, err := m.buildDistillForwardTape(in, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:6], false, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodeDistillForwardBoundary(out, in)
	if err != nil {
		return DistillForwardResult{}, b.fail(err)
	}
	return r, nil
}

// Sources must pass complete tape/binding admission before this synchronous
// remap. The caller admits combined storage/bindings after composition.
func appendAdmittedFlowGraph(t *train.Tape, p FlowDualTrainingTape, input int, parameters map[string]train.ResidentTapeBinding) ([]int, map[string]train.ResidentTapeBinding, error) {
	if t == nil || input < 0 || input >= len(t.Tensors) || p.inputValue < 0 || p.inputValue >= len(p.Tape.Tensors) || len(p.Tape.Tensors[p.inputValue].Values) == 0 {
		return nil, nil, fmt.Errorf("distill forward input ID")
	}
	if parameters != nil {
		if len(parameters) != len(p.ResidentBindings) {
			return nil, nil, fmt.Errorf("distill forward parameter inventory")
		}
		for _, b := range parameters {
			if b.Parameter < 0 || b.Parameter >= len(t.Tensors) {
				return nil, nil, fmt.Errorf("distill forward parameter ID")
			}
		}
	}
	offset := len(t.Tensors)
	t.Tensors = append(t.Tensors, p.Tape.Tensors...)
	mapping := make([]int, len(p.Tape.Tensors))
	for i := range mapping {
		mapping[i] = offset + i
	}
	mapping[p.inputValue] = input
	a, _ := p.Tape.Tensors[p.inputValue].Elements()
	z, _ := t.Tensors[input].Elements()
	if a != z {
		return nil, nil, fmt.Errorf("distill forward input edge")
	}
	for name, b := range p.ResidentBindings {
		if parameters != nil {
			e, ok := parameters[name]
			if !ok {
				return nil, nil, fmt.Errorf("distill forward parameter inventory")
			}
			a, _ := p.Tape.Tensors[b.Parameter].Elements()
			z, _ := t.Tensors[e.Parameter].Elements()
			if a != z {
				return nil, nil, fmt.Errorf("distill forward parameter extent")
			}
			mapping[b.Parameter] = e.Parameter
		}
	}
	// Remapped operands own disjoint, capacity-limited regions. Each append
	// invocation owns its arena; neither source nor sibling appends share it.
	var operands flowOperandArena
	for _, c := range p.Tape.Commands {
		c.Inputs = operands.copy(c.Inputs)
		for i, id := range c.Inputs {
			c.Inputs[i] = mapping[id]
		}
		c.Output = mapping[c.Output]
		t.Commands = append(t.Commands, c)
	}
	bindings := map[string]train.ResidentTapeBinding{}
	for name, b := range p.ResidentBindings {
		bindings[name] = train.ResidentTapeBinding{Parameter: mapping[b.Parameter], Gradient: mapping[b.Gradient]}
	}
	outputs := make([]int, len(p.Tape.Outputs))
	for i, id := range p.Tape.Outputs {
		outputs[i] = mapping[id]
	}
	return outputs, bindings, nil
}
