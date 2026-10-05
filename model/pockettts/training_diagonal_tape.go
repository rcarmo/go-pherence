package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingDiagonalBackend combines diagonal LSD seeds/scaling, flow backward and
// direct target assembly. The forward prediction remains an admission boundary.
type TrainingDiagonalBackend interface {
	DiagonalEnabled() bool
	EvaluateDiagonal(*FlowHeadCPU, []float32, float32, []float32, []float32, []float32, float32, float32) (DiagonalTrainingResult, error)
}
type DiagonalTrainingResult struct {
	DCondition, DTarget []float32
	RawSquare, Loss     float64
	DLogVariance        float32 // unscaled; step reduction applies the objective once
}
type DiagonalTrainingTape struct {
	Tape                             train.Tape
	ResidentBindings                 map[string]train.ResidentTapeBinding
	inputID, desiredID, predictionID int // private retained-forward edges
}

func (b *ResidentTrainingBackend) DiagonalEnabled() bool {
	return b != nil && (b.EndpointEnabled() || b.Scope == "ptx-resident-diagfused-hybrid-v1" || b.Scope == "vulkan-resident-diagfused-hybrid-v1")
}

// BuildDiagonalTrainingTape accepts a forward prediction only for conservative
// loss overflow admission; flow forward is recomputed inside the VJP graph.
func (m *FlowHeadCPU) BuildDiagonalTrainingTape(condition []float32, time float32, input, prediction, desired []float32, logvar, scale float32, maxBytes int64) (DiagonalTrainingTape, error) {
	return m.buildDiagonalTrainingTape(condition, time, input, prediction, desired, logvar, scale, maxBytes, false)
}
func (m *FlowHeadCPU) buildDiagonalTrainingTape(condition []float32, time float32, input, prediction, desired []float32, logvar, scale float32, maxBytes int64, borrow bool) (DiagonalTrainingTape, error) {
	var result DiagonalTrainingTape
	if !isFinite(time) || time < 0 || time > 1 || !isFinite(scale) || scale < 0 || scale > 1 {
		return result, fmt.Errorf("diagonal time/scale")
	}
	if err := admitLSDLossTraining(prediction, desired, logvar, maxBytes); err != nil {
		return result, err
	}
	n := len(prediction)
	var g *flowTapeGraph
	stats, scaledSeed := -1, -1
	desiredID, predictionID := -1, -1
	p, err := m.buildFlowDualTrainingTapeSeed(condition, []float32{time, time}, input, -1, make([]float32, n), make([]float32, n), maxBytes, true, borrow, func(graph *flowTapeGraph, out graphDual) (int, int) {
		g = graph
		target := g.constant(desired)
		desiredID, predictionID = target, out.v
		lv := g.constant([]float32{logvar})
		stats = g.command("lsd-loss", []int{out.v, target, lv}, []int{3}, 0, 0, 0, 0, 0)
		seed := g.command("lsd-loss", []int{out.v, target, lv}, []int{n}, 0, 1, 0, 0, 0)
		scaledSeed = g.command("vector", []int{g.zero(n), seed}, []int{n}, 0, 0, 0, scale, 0)
		return scaledSeed, g.zero(n)
	})
	if err != nil {
		return result, err
	}
	// Existing diagonal-target order is time*dX - already-scaled prediction seed.
	scaledInput := g.command("vector", []int{g.zero(n), p.Tape.Outputs[3]}, []int{n}, 0, 0, 0, time, 0)
	dtarget := g.command("vector", []int{scaledInput, scaledSeed}, []int{n}, 0, 0, 0, -1, 0)
	g.tape.Outputs = []int{p.Tape.Outputs[2], dtarget, stats}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	return DiagonalTrainingTape{Tape: g.tape, ResidentBindings: p.ResidentBindings, inputID: p.inputValue, desiredID: desiredID, predictionID: predictionID}, nil
}
func validateDiagonalResult(r DiagonalTrainingResult, condition, input, prediction, desired []float32) error {
	if len(r.DCondition) != len(condition) || len(r.DTarget) != len(input) || !finiteF32(r.DCondition) || !finiteF32(r.DTarget) || !finite64(r.RawSquare) || r.RawSquare < 0 || !finite64(r.Loss) || !isFinite(r.DLogVariance) {
		return fmt.Errorf("diagonal result shape/nonfinite/square")
	}
	if f32SlicesOverlap(r.DCondition, r.DTarget) {
		return fmt.Errorf("diagonal aliased outputs")
	}
	for _, source := range [][]float32{condition, input, prediction, desired} {
		if f32SlicesOverlap(r.DCondition, source) || f32SlicesOverlap(r.DTarget, source) {
			return fmt.Errorf("diagonal borrowed output")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateDiagonal(m *FlowHeadCPU, condition []float32, time float32, input, prediction, desired []float32, logvar, scale float32) (DiagonalTrainingResult, error) {
	var r DiagonalTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.DiagonalEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.LossBackend != b || m.BranchBackend != b || m.StepReductionBackend != b {
		return r, b.fail(fmt.Errorf("diagonal owner/scope"))
	}
	if b.deviceHandoffEnabled() && b.handoffLease != nil {
		return b.evaluateDeviceDiagonal(m, condition, time, input, prediction, desired, logvar, scale)
	}
	p, err := m.buildDiagonalTrainingTape(condition, time, input, prediction, desired, logvar, scale, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runPredictionBackward(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], condition, "diagonal", [][]float32{input, desired, prediction}, []int{p.inputID, p.desiredID, p.predictionID}, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodeDiagonalBoundary(out, condition, input, prediction, desired)
	if err != nil {
		return DiagonalTrainingResult{}, b.fail(err)
	}
	if err := b.closePredictionHandoff(); err != nil {
		return DiagonalTrainingResult{}, err
	}
	return r, nil
}
