package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// PrimaryTrainingInputs are raw mixed-AD seeds and endpoint input-only VJP.
// Parameter objective scaling is separate; returned input derivatives stay raw.
type PrimaryTrainingInputs struct {
	Condition, Input, GV, GT, EndpointInput, Target, Noise []float32
	S, T, DDelta                                           float32
}
type PrimaryTrainingResult struct{ DCondition, DTarget, DNoise, DTimes []float32 }
type PrimaryTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
	// Explicit composition edges: value/tangent seeds, endpoint input VJP, delta.
	seedIDs [4]int
	inputID int
}
type TrainingPrimaryBackend interface {
	PrimaryEnabled() bool
	EvaluatePrimary(*FlowHeadCPU, PrimaryTrainingInputs, float32) (PrimaryTrainingResult, error)
}

func (b *ResidentTrainingBackend) PrimaryEnabled() bool {
	return b != nil && (b.DistillEnabled() || b.Scope == "ptx-resident-primaryfused-hybrid-v1" || b.Scope == "vulkan-resident-primaryfused-hybrid-v1")
}
func (m *FlowHeadCPU) BuildPrimaryTrainingTape(input PrimaryTrainingInputs, maxBytes int64) (PrimaryTrainingTape, error) {
	return m.buildPrimaryTrainingTape(input, maxBytes, false)
}
func (m *FlowHeadCPU) buildPrimaryTrainingTape(in PrimaryTrainingInputs, maxBytes int64, borrow bool) (PrimaryTrainingTape, error) {
	var result PrimaryTrainingTape
	if m == nil || len(m.Time) != 2 || !isFinite(in.S) || !isFinite(in.T) || in.S < 0 || in.T > 1 || in.S > in.T || !isFinite(in.DDelta) {
		return result, fmt.Errorf("primary time/topology/delta")
	}
	n := len(in.Input)
	for _, x := range [][]float32{in.EndpointInput, in.Target, in.Noise} {
		if len(x) != n || !finiteF32(x) {
			return result, fmt.Errorf("primary finish input shape/nonfinite")
		}
	}
	var g *flowTapeGraph
	var seedIDs [4]int
	p, err := m.buildFlowDualTrainingTapeSeed(in.Condition, []float32{in.S, in.T}, in.Input, 1, in.GV, in.GT, maxBytes, true, borrow, func(graph *flowTapeGraph, _ graphDual) (int, int) {
		g = graph
		seedIDs[0], seedIDs[1] = g.constant(in.GV), g.constant(in.GT)
		return seedIDs[0], seedIDs[1]
	})
	if err != nil {
		return result, err
	}
	add := func(a, b int, scale float32) int {
		return g.command("vector", []int{a, b}, []int{g.elements(a)}, 0, 0, 0, scale, 0)
	}
	scale := func(a int, s float32) int { return add(g.zero(g.elements(a)), a, s) }
	dot := func(a, b int) int {
		return g.reshape(g.command("linear-forward", []int{g.reshape(a, 1, n), g.reshape(g.zero(n), 1, n), g.reshape(b, 1, n), g.zero(1)}, []int{1, 1}, 0, 0, 0, 0, 0), 1)
	}
	seedIDs[2] = g.constant(in.EndpointInput)
	dx := add(p.Tape.Outputs[3], seedIDs[2], 1)
	target, noise := scale(dx, in.S), scale(dx, 1-in.S)
	difference := add(g.constant(in.Target), g.constant(in.Noise), -1)
	dd := g.constant([]float32{in.DDelta})
	seedIDs[3] = dd
	ds := add(add(dot(dx, difference), p.Tape.Outputs[4], 1), dd, -1)
	dt := add(p.Tape.Outputs[5], dd, 1)
	pad := func(a, offset int) int { return g.command("vector", []int{a, g.zero(1)}, []int{2}, 0, 2, offset, 0, 0) }
	times := add(pad(ds, 0), pad(dt, 1), 1)
	g.tape.Outputs = []int{p.Tape.Outputs[2], target, noise, times}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	return PrimaryTrainingTape{Tape: g.tape, ResidentBindings: p.ResidentBindings, seedIDs: seedIDs, inputID: p.inputValue}, nil
}
func validatePrimaryResult(r PrimaryTrainingResult, in PrimaryTrainingInputs) error {
	values := [][]float32{r.DCondition, r.DTarget, r.DNoise, r.DTimes}
	sizes := []int{len(in.Condition), len(in.Input), len(in.Input), 2}
	for i, x := range values {
		if len(x) != sizes[i] || !finiteF32(x) {
			return fmt.Errorf("primary result shape/nonfinite")
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("primary aliased outputs")
			}
		}
		for _, source := range [][]float32{in.Condition, in.Input, in.GV, in.GT, in.EndpointInput, in.Target, in.Noise} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("primary borrowed output")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluatePrimary(m *FlowHeadCPU, in PrimaryTrainingInputs, parameterScale float32) (PrimaryTrainingResult, error) {
	var r PrimaryTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.PrimaryEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("primary owner/scope"))
	}
	if !isFinite(parameterScale) || parameterScale < 0 {
		return r, b.fail(fmt.Errorf("primary parameter scale"))
	}
	p, err := m.buildPrimaryTrainingTape(in, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:4], true, parameterScale)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodePrimaryBoundary(out, in)
	if err != nil {
		return PrimaryTrainingResult{}, b.fail(err)
	}
	return r, nil
}
