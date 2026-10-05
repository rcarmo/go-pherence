package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingEndpointBackend fuses minimal-LSD loss and endpoint input-only VJP.
// Endpoint parameter/condition/time contributions must never accumulate.
type TrainingEndpointBackend interface {
	EndpointEnabled() bool
	EvaluateEndpoint(*FlowHeadCPU, []float32, float32, []float32, []float32, []float32, []float32, []float32, float32, float32) (EndpointTrainingResult, error)
}
type EndpointTrainingResult struct {
	DInput, GV, GT       []float32
	RawSquare, Loss      float64
	DLogVariance, DDelta float32
}
type EndpointTrainingTape struct {
	Tape                                                    train.Tape
	ResidentBindings                                        map[string]train.ResidentTapeBinding
	inputID, desiredID, predictionID, velocityID, tangentID int
}

func (b *ResidentTrainingBackend) EndpointEnabled() bool {
	return b != nil && (b.PrimaryEnabled() || b.Scope == "ptx-resident-endpointfused-hybrid-v1" || b.Scope == "vulkan-resident-endpointfused-hybrid-v1")
}

// Endpoint prediction is used only for the existing conservative overflow gate.
// The immutable endpoint is reevaluated in graph. All parameter bindings remain
// admitted even though resident execution explicitly disables accumulation.
func (m *FlowHeadCPU) BuildEndpointTrainingTape(condition []float32, time float32, input, prediction, desired, velocity, tangent []float32, logvar, delta float32, maxBytes int64) (EndpointTrainingTape, error) {
	return m.buildEndpointTrainingTape(condition, time, input, prediction, desired, velocity, tangent, logvar, delta, maxBytes, false)
}
func (m *FlowHeadCPU) buildEndpointTrainingTape(condition []float32, time float32, input, prediction, desired, velocity, tangent []float32, logvar, delta float32, maxBytes int64, borrow bool) (EndpointTrainingTape, error) {
	var result EndpointTrainingTape
	if !isFinite(time) || time < 0 || time > 1 || !isFinite(delta) || delta < 0 || delta > time {
		return result, fmt.Errorf("endpoint time/delta")
	}
	if err := admitLSDLossTraining(desired, prediction, logvar, maxBytes); err != nil {
		return result, err
	}
	n := len(prediction)
	if len(velocity) != n || len(tangent) != n || !finiteF32(velocity) || !finiteF32(tangent) {
		return result, fmt.Errorf("endpoint primary boundaries")
	}
	var g *flowTapeGraph
	stats, dr := -1, -1
	desiredID, predictionID := -1, -1
	p, err := m.buildFlowDualTrainingTapeSeed(condition, []float32{time, time}, input, -1, make([]float32, n), make([]float32, n), maxBytes, true, borrow, func(graph *flowTapeGraph, out graphDual) (int, int) {
		g = graph
		desiredID = g.constant(desired)
		predictionID = out.v
		lv := g.constant([]float32{logvar})
		stats = g.command("lsd-loss", []int{desiredID, out.v, lv}, []int{3}, 0, 0, 0, 0, 0)
		dr = g.command("lsd-loss", []int{desiredID, out.v, lv}, []int{n}, 0, 1, 0, 0, 0)
		de := g.command("vector", []int{g.zero(n), dr}, []int{n}, 0, 0, 0, -1, 0)
		return de, g.zero(n)
	})
	if err != nil {
		return result, err
	}
	dx := p.Tape.Outputs[3]
	gv := g.command("vector", []int{dr, dx}, []int{n}, 0, 0, 0, delta, 0)
	gt := g.command("vector", []int{g.zero(n), dr}, []int{n}, 0, 0, 0, delta, 0)
	dot := func(a, b int) int {
		return g.reshape(g.command("linear-forward", []int{g.reshape(a, 1, n), g.reshape(g.zero(n), 1, n), g.reshape(b, 1, n), g.zero(1)}, []int{1, 1}, 0, 0, 0, 0, 0), 1)
	}
	tangentID := g.constant(tangent)
	dt := dot(dr, tangentID)
	velocityID := g.constant(velocity)
	dv := dot(dx, velocityID)
	dd := g.command("vector", []int{dt, dv}, []int{1}, 0, 0, 0, 1, 0)
	g.tape.Outputs = []int{dx, gv, gt, stats, dd}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	return EndpointTrainingTape{Tape: g.tape, ResidentBindings: p.ResidentBindings, inputID: p.inputValue, desiredID: desiredID, predictionID: predictionID, velocityID: velocityID, tangentID: tangentID}, nil
}
func validateEndpointResult(r EndpointTrainingResult, condition, input, prediction, desired, velocity, tangent []float32) error {
	n := len(input)
	if len(r.DInput) != n || len(r.GV) != n || len(r.GT) != n || !finiteF32(r.DInput) || !finiteF32(r.GV) || !finiteF32(r.GT) || !finite64(r.RawSquare) || r.RawSquare < 0 || !finite64(r.Loss) || !isFinite(r.DLogVariance) || !isFinite(r.DDelta) {
		return fmt.Errorf("endpoint result shape/nonfinite")
	}
	values := [][]float32{r.DInput, r.GV, r.GT}
	for i, x := range values {
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("endpoint aliased outputs")
			}
		}
		for _, source := range [][]float32{condition, input, prediction, desired, velocity, tangent} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("endpoint borrowed output")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateEndpoint(m *FlowHeadCPU, condition []float32, time float32, input, prediction, desired, velocity, tangent []float32, logvar, delta float32) (EndpointTrainingResult, error) {
	var r EndpointTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.EndpointEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.LossBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("endpoint owner/scope"))
	}
	p, err := m.buildEndpointTrainingTape(condition, time, input, prediction, desired, velocity, tangent, logvar, delta, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	// This false is the minimal-LSD stop-gradient contract, including when the
	// surrounding objective weight is nonzero. Pruning retains input-VJP roots.
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:5], false, 1)
	if err != nil {
		return r, b.fail(err)
	}
	if len(out) != 5 || len(out[3]) != 3 || len(out[4]) != 1 {
		return r, b.fail(fmt.Errorf("endpoint boundary count"))
	}
	r = EndpointTrainingResult{out[0], out[1], out[2], float64(out[3][0]), float64(out[3][1]), out[3][2], out[4][0]}
	if err := validateEndpointResult(r, condition, input, prediction, desired, velocity, tangent); err != nil {
		return EndpointTrainingResult{}, b.fail(err)
	}
	return r, nil
}
