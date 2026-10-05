package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

// DistillTrainingInputs retain immutable forward values for overflow admission
// and the minimal-LSD stop-gradient policy. Gradient handoffs stay in the tape.
type DistillTrainingInputs struct {
	Condition, PrimaryInput, EndpointInput, Prediction, Desired, Velocity, Tangent, Target, Noise []float32
	S, T, LogVariance                                                                             float32
}
type DistillTrainingResult struct {
	PrimaryTrainingResult
	RawSquare, Loss float64
	DLogVariance    float32
}
type DistillTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
	forwardIDs       [6]int // primary input, velocity, tangent, endpoint input, desired, prediction
}
type TrainingDistillBackend interface {
	DistillEnabled() bool
	EvaluateDistill(*FlowHeadCPU, DistillTrainingInputs, float32) (DistillTrainingResult, error)
}

func (b *ResidentTrainingBackend) DistillEnabled() bool {
	return b != nil && (b.DistillForwardEnabled() || b.Scope == "ptx-resident-distillfused-hybrid-v1" || b.Scope == "vulkan-resident-distillfused-hybrid-v1")
}
func (m *FlowHeadCPU) BuildDistillTrainingTape(in DistillTrainingInputs, maxBytes int64) (DistillTrainingTape, error) {
	return m.buildDistillTrainingTape(in, maxBytes, false)
}
func (m *FlowHeadCPU) buildDistillTrainingTape(in DistillTrainingInputs, maxBytes int64, borrow bool) (DistillTrainingTape, error) {
	return m.buildScaledDistillTrainingTape(in, maxBytes, borrow, 1)
}

// Apply the objective to primary adjoints before reverse, matching CPU training.
// Loss statistics stay raw for the weighting/reduction contract.
func (m *FlowHeadCPU) buildScaledDistillTrainingTape(in DistillTrainingInputs, maxBytes int64, borrow bool, scale float32) (DistillTrainingTape, error) {
	var result DistillTrainingTape
	if !isFinite(in.S) || !isFinite(in.T) || in.S < 0 || in.T > 1 || in.S > in.T {
		return result, fmt.Errorf("distill time")
	}
	endpoint, err := m.buildEndpointTrainingTape(in.Condition, in.T, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.LogVariance, in.T-in.S, maxBytes, borrow)
	if err != nil {
		return result, err
	}
	n := len(in.Prediction)
	primary, err := m.buildPrimaryTrainingTape(PrimaryTrainingInputs{Condition: in.Condition, Input: in.PrimaryInput, GV: make([]float32, n), GT: make([]float32, n), EndpointInput: make([]float32, n), Target: in.Target, Noise: in.Noise, S: in.S, T: in.T}, maxBytes, borrow)
	if err != nil {
		return result, err
	}
	return composeScaledDistillTrainingTapes(endpoint, primary, maxBytes, scale)
}

func composeScaledDistillTrainingTapes(endpoint EndpointTrainingTape, primary PrimaryTrainingTape, maxBytes int64, scale float32) (DistillTrainingTape, error) {
	if !isFinite(scale) || scale < 0 {
		return DistillTrainingTape{}, fmt.Errorf("distill objective scale")
	}
	if err := train.ValidateResidentTapeBindings(endpoint.Tape, endpoint.ResidentBindings, maxBytes); err != nil {
		return DistillTrainingTape{}, err
	}
	if len(endpoint.Tape.Outputs) < 5 {
		return DistillTrainingTape{}, fmt.Errorf("distill source outputs")
	}
	if scale != 1 {
		g := &flowTapeGraph{tape: endpoint.Tape}
		g.tape.Tensors = append([]train.TensorSpec(nil), endpoint.Tape.Tensors...)
		g.tape.Commands = append([]train.Command(nil), endpoint.Tape.Commands...)
		g.tape.Outputs = append([]int(nil), endpoint.Tape.Outputs...)
		for _, slot := range []int{0, 1, 2, 4} {
			id := g.tape.Outputs[slot]
			n, err := g.tape.Tensors[id].Elements()
			if err != nil {
				return DistillTrainingTape{}, err
			}
			g.tape.Outputs[slot] = g.command("vector", []int{g.zero(n), id}, []int{n}, 0, 0, 0, scale, 0)
		}
		endpoint.Tape = g.tape
	}
	return composeDistillTrainingTapes(endpoint, primary, maxBytes)
}

// Admit complete sources before remapping any edge. The helper accepts already
// constructed component graphs so topology incompatibilities are testable
// without test-only builder hooks or a change to their numerical contract.
func composeDistillTrainingTapes(endpoint EndpointTrainingTape, primary PrimaryTrainingTape, maxBytes int64) (DistillTrainingTape, error) {
	var result DistillTrainingTape
	if err := train.ValidateResidentTapeBindings(primary.Tape, primary.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	// Admit each complete source before connecting graphs; combined logical
	// storage still includes dead placeholders/duplicate parameter sources.
	if err := train.ValidateResidentTapeBindings(endpoint.Tape, endpoint.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	if len(endpoint.Tape.Outputs) < 5 || len(primary.Tape.Outputs) < 4 {
		return result, fmt.Errorf("distill source outputs")
	}
	tensors := len(endpoint.Tape.Tensors) + len(primary.Tape.Tensors)
	commands := len(endpoint.Tape.Commands) + len(primary.Tape.Commands)
	if tensors > 4096 || commands > 4096 {
		return result, fmt.Errorf("distill combined graph ceiling")
	}
	t := endpoint.Tape
	// Complete source extents are known. Own one final metadata backing array
	// instead of cloning then repeatedly growing it; nested source arrays stay
	// read-only and remapped command inputs below retain their owned copies.
	t.Tensors = make([]train.TensorSpec, len(endpoint.Tape.Tensors), tensors)
	copy(t.Tensors, endpoint.Tape.Tensors)
	t.Commands = make([]train.Command, len(endpoint.Tape.Commands), commands)
	copy(t.Commands, endpoint.Tape.Commands)
	offset := len(t.Tensors)
	t.Tensors = append(t.Tensors, primary.Tape.Tensors...)
	mapping := make([]int, len(primary.Tape.Tensors))
	for i := range mapping {
		mapping[i] = offset + i
	}
	// Bind one live parameter ID for both forward graphs, but expose only the
	// primary gradient. Endpoint parameter gradients are dead, never added.
	if len(primary.ResidentBindings) != len(endpoint.ResidentBindings) {
		return result, fmt.Errorf("distill parameter inventory")
	}
	for name, b := range primary.ResidentBindings {
		e, ok := endpoint.ResidentBindings[name]
		if !ok {
			return result, fmt.Errorf("distill missing endpoint parameter %s", name)
		}
		a, _ := primary.Tape.Tensors[b.Parameter].Elements()
		z, _ := endpoint.Tape.Tensors[e.Parameter].Elements()
		if a != z {
			return result, fmt.Errorf("distill parameter extent")
		}
		mapping[b.Parameter] = e.Parameter
	}
	for i, source := range []int{endpoint.Tape.Outputs[1], endpoint.Tape.Outputs[2], endpoint.Tape.Outputs[0], endpoint.Tape.Outputs[4]} {
		id := primary.seedIDs[i]
		if id < 0 || id >= len(mapping) {
			return result, fmt.Errorf("distill seed edge")
		}
		a, _ := primary.Tape.Tensors[id].Elements()
		z, _ := endpoint.Tape.Tensors[source].Elements()
		if a != z {
			return result, fmt.Errorf("distill seed extent")
		}
		mapping[id] = source
	}
	// Own remapped operand regions without one allocation per command. The
	// arena is local to this admitted composition, never shared with sources.
	var operands flowOperandArena
	for _, c := range primary.Tape.Commands {
		c.Inputs = operands.copy(c.Inputs)
		for i, id := range c.Inputs {
			c.Inputs[i] = mapping[id]
		}
		c.Output = mapping[c.Output]
		t.Commands = append(t.Commands, c)
	}
	bindings := map[string]train.ResidentTapeBinding{}
	t.Outputs = make([]int, 0, 5+len(primary.ResidentBindings))
	for _, id := range primary.Tape.Outputs[:4] {
		t.Outputs = append(t.Outputs, mapping[id])
	}
	t.Outputs = append(t.Outputs, endpoint.Tape.Outputs[3])
	names := make([]string, 0, len(primary.ResidentBindings))
	for name := range primary.ResidentBindings {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		b := primary.ResidentBindings[name]
		binding := train.ResidentTapeBinding{Parameter: mapping[b.Parameter], Gradient: mapping[b.Gradient]}
		bindings[name] = binding
		t.Outputs = append(t.Outputs, binding.Gradient)
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return result, err
	}
	return DistillTrainingTape{Tape: t, ResidentBindings: bindings, forwardIDs: [6]int{mapping[primary.inputID], endpoint.velocityID, endpoint.tangentID, endpoint.inputID, endpoint.desiredID, endpoint.predictionID}}, nil
}
func validateDistillResult(r DistillTrainingResult, in DistillTrainingInputs) error {
	if !finite64(r.RawSquare) || r.RawSquare < 0 || !finite64(r.Loss) || !isFinite(r.DLogVariance) {
		return fmt.Errorf("distill loss stats")
	}
	if err := validatePrimaryResult(r.PrimaryTrainingResult, PrimaryTrainingInputs{Condition: in.Condition, Input: in.PrimaryInput, Target: in.Target, Noise: in.Noise}); err != nil {
		return err
	}
	for _, x := range [][]float32{r.DCondition, r.DTarget, r.DNoise, r.DTimes} {
		for _, source := range [][]float32{in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("distill borrowed endpoint")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateDistill(m *FlowHeadCPU, in DistillTrainingInputs, parameterScale float32) (DistillTrainingResult, error) {
	var r DistillTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.DistillEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.LossBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("distill owner/scope"))
	}
	if !isFinite(parameterScale) || parameterScale < 0 {
		return r, b.fail(fmt.Errorf("distill parameter scale"))
	}
	p, err := m.buildScaledDistillTrainingTape(in, b.MaxTapeBytes, true, parameterScale)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:5], true, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodeDistillBoundary(out, in)
	if err != nil {
		return DistillTrainingResult{}, b.fail(err)
	}
	return r, nil
}
