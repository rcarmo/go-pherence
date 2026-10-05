package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Step-only result: no noise/time derivative placeholders escape this boundary.
// Condition/target adjoints are objective-scaled before primary reverse.
// Loss/logvar statistics stay raw; step reduction scales logvar once.
type stepDistillResult struct {
	DCondition, DTarget []float32
	RawSquare, Loss     float64
	DLogVariance        float32
}

func (b *ResidentTrainingBackend) distillResultsEnabled() bool {
	return b != nil && (b.eosForwardEnabled() || b.Scope == "ptx-resident-distillresults-hybrid-v1" || b.Scope == "vulkan-resident-distillresults-hybrid-v1")
}

// Retain complete source admission and the finite checks of the discarded
// public noise/time outputs. Runtime zero-weight dot products propagate NaN/Inf
// without an overflow-prone sum of finite derivatives. Their scalar sum is zero
// iff those derivatives are finite (IEEE F32; no fast-math zero elimination).
// These roots also retain the original derivative computation under pruning.
func buildDistillResultsTape(p DistillTrainingTape, maxBytes int64) (DistillTrainingTape, error) {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return DistillTrainingTape{}, err
	}
	if len(p.Tape.Outputs) < 5 {
		return DistillTrainingTape{}, fmt.Errorf("distill result source count")
	}
	g := &transformerTapeGraph{constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, borrowParameters: true}
	g.tape = p.Tape
	g.tape.Tensors = append([]train.TensorSpec(nil), p.Tape.Tensors...)
	g.tape.Commands = append([]train.Command(nil), p.Tape.Commands...)
	g.tape.Outputs = append([]int(nil), p.Tape.Outputs...)
	witness := appendTrainingFiniteWitness(g, p.Tape.Outputs[2:4])
	g.tape.Outputs = append(g.tape.Outputs, witness)
	built, err := materializeSequenceTape(g, p.ResidentBindings, maxBytes)
	if err != nil {
		return DistillTrainingTape{}, err
	}
	return DistillTrainingTape{Tape: built.Tape, ResidentBindings: built.ResidentBindings, forwardIDs: p.forwardIDs}, nil
}

// Inputs are validated graph tensor IDs. Runtime zero weights preserve IEEE
// NaN/Inf propagation without reducing finite magnitudes into an overflow.
func appendTrainingFiniteWitness(g *transformerTapeGraph, ids []int) int {
	var witness int
	first := true
	for _, id := range ids {
		n, _ := g.tape.Tensors[id].Elements()
		// LinearShape limits K to 16384. Preserve the small-source command order;
		// large admitted arrays use bounded slices, never an oversized dot.
		for offset := 0; offset < n; offset += 16384 {
			size := min(n-offset, 16384)
			source := id
			if n > 16384 {
				source = g.slice(id, offset, size)
			}
			x := g.reshape(source, 1, size)
			dot := g.command("linear-forward", []int{x, g.zero(1, size), g.zero(1, size), g.zero(1)}, []int{1, 1}, 0)
			next := g.flat(dot)
			if first {
				witness = next
				first = false
			} else {
				witness = g.command("vector", []int{witness, next}, []int{1}, 0)
				g.tape.Commands[len(g.tape.Commands)-1].Scale = 1
			}
		}
	}
	return witness
}
func distillResultsOutputs(p DistillTrainingTape) []int {
	return []int{p.Tape.Outputs[0], p.Tape.Outputs[1], p.Tape.Outputs[4], p.Tape.Outputs[len(p.Tape.Outputs)-1]}
}
func decodeDistillResults(out [][]float32, in DistillTrainingInputs) (stepDistillResult, error) {
	var r stepDistillResult
	if len(out) != 4 {
		return r, fmt.Errorf("distill private boundary count")
	}
	sizes := []int{len(in.Condition), len(in.PrimaryInput), 3, 1}
	for i, x := range out {
		if len(x) != sizes[i] || !finiteF32(x) {
			return r, fmt.Errorf("distill private shape/nonfinite")
		}
		for _, y := range out[:i] {
			if f32SlicesOverlap(x, y) {
				return r, fmt.Errorf("distill private aliased outputs")
			}
		}
		for _, source := range [][]float32{in.Condition, in.PrimaryInput, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.Target, in.Noise} {
			if f32SlicesOverlap(x, source) {
				return r, fmt.Errorf("distill private borrowed output")
			}
		}
	}
	if out[2][0] < 0 || out[3][0] != 0 {
		return r, fmt.Errorf("distill private stats/witness")
	}
	return stepDistillResult{out[0], out[1], float64(out[2][0]), float64(out[2][1]), out[2][2]}, nil
}
func (b *ResidentTrainingBackend) evaluateDistillResults(m *FlowHeadCPU, in DistillTrainingInputs, parameterScale float32) (stepDistillResult, error) {
	var r stepDistillResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.distillResultsEnabled() || b.trainer.Flow != m || m.TrainingBackend != b || m.LossBackend != b || m.BranchBackend != b {
		return r, b.fail(fmt.Errorf("distill private owner/scope"))
	}
	if !isFinite(parameterScale) || parameterScale < 0 {
		return r, b.fail(fmt.Errorf("distill private parameter scale"))
	}
	if b.deviceHandoffEnabled() && b.handoffLease != nil {
		return b.evaluateDeviceDistill(m, in, parameterScale)
	}
	p, err := m.buildScaledDistillTrainingTape(in, b.MaxTapeBytes, true, parameterScale)
	if err != nil {
		return r, b.fail(err)
	}
	p, err = buildDistillResultsTape(p, b.MaxTapeBytes)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runPredictionBackward(p.Tape, p.ResidentBindings, distillResultsOutputs(p), in.Condition, "distill", [][]float32{in.PrimaryInput, in.Velocity, in.Tangent, in.EndpointInput, in.Desired, in.Prediction}, p.forwardIDs[:], 1)
	if err != nil {
		return r, b.fail(err)
	}
	r, err = decodeDistillResults(out, in)
	if err != nil {
		return stepDistillResult{}, b.fail(err)
	}
	if err := b.closePredictionHandoff(); err != nil {
		return stepDistillResult{}, err
	}
	return r, nil
}
