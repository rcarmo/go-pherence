package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (b *ResidentTrainingBackend) predictionHandoffEnabled() bool {
	return b != nil && (b.Scope == "ptx-resident-predictionhandoff-hybrid-v1" || b.Scope == "vulkan-resident-predictionhandoff-hybrid-v1")
}

// Retain every original forward output plus guard for host admission. The first
// sliceCount roots also become explicit read-only backward inputs. No implicit
// gradients cross these edges; backward retains the original adjoint graph.
func (b *ResidentTrainingBackend) beginPredictionHandoff(t train.Tape, bindings map[string]train.ResidentTapeBinding, outputs []int, condition []float32, kind string, sliceCount int) ([][]float32, error) {
	if b.predictionLease != nil || b.handoffLease == nil || !b.predictionHandoffEnabled() || sliceCount <= 0 || sliceCount > len(outputs) {
		return nil, fmt.Errorf("prediction handoff owner/already live")
	}
	var err error
	var edges []train.TapeHandoffEdge
	t, edges, err = b.bindFlowHandoffCondition(t, bindings, condition)
	if err != nil {
		return nil, err
	}
	sources := make([]train.ResidentLeaseEdge, len(edges))
	for i, e := range edges {
		sources[i] = train.ResidentLeaseEdge{Source: b.handoffLease, Output: e.Output, Input: e.Input}
	}
	b.predictionLease, err = b.handoffLease.BeginChildTape(b.ctx, t, bindings, outputs, sources)
	if err != nil {
		return nil, err
	}
	values, err := b.predictionLease.DownloadOutputs(b.ctx, outputs)
	if err != nil {
		return nil, err
	}
	if sliceCount > len(values) {
		return nil, fmt.Errorf("prediction roots count")
	}
	b.predictionValues = values[:sliceCount]
	b.predictionRoots = append([]int(nil), outputs[:sliceCount]...)
	b.predictionKind = kind
	b.predictionRow = b.handoffRow
	return values, nil
}

// Replace explicit forward-value reads while preserving commands/adjoint paths.
// Computed prediction destinations are never rebound or dropped: only their
// consumers read a borrowed placeholder, and all original commands stay admitted.
func (b *ResidentTrainingBackend) runPredictionBackward(t train.Tape, bindings map[string]train.ResidentTapeBinding, outputs []int, condition []float32, kind string, values [][]float32, ids []int, scale float32) ([][]float32, error) {
	if b.predictionLease == nil {
		return b.runResidentTape(t, bindings, outputs, true, scale)
	}
	if !b.predictionHandoffEnabled() || b.predictionKind != kind || b.predictionRow != b.handoffRow || len(values) != len(b.predictionValues) || len(ids) != len(values) || len(b.predictionRoots) != len(values) {
		return nil, fmt.Errorf("prediction handoff backward provenance")
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, b.MaxTapeBytes); err != nil {
		return nil, err
	}
	for i, id := range ids {
		if id < 0 || id >= len(t.Tensors) || len(values[i]) == 0 || len(values[i]) != len(b.predictionValues[i]) || &values[i][0] != &b.predictionValues[i][0] {
			return nil, fmt.Errorf("prediction handoff borrowed host provenance")
		}
		n, _ := t.Tensors[id].Elements()
		if n != len(values[i]) {
			return nil, fmt.Errorf("prediction edge extent")
		}
		for _, binding := range bindings {
			if id == binding.Parameter {
				return nil, fmt.Errorf("prediction edge parameter alias")
			}
		}
		if constants := t.Tensors[id].Values; constants != nil {
			for j, v := range constants {
				if v != values[i][j] {
					return nil, fmt.Errorf("prediction edge constant provenance")
				}
			}
		}
	}
	t, conditionEdges, err := b.bindFlowHandoffCondition(t, bindings, condition)
	if err != nil {
		return nil, err
	}
	sources := make([]train.ResidentLeaseEdge, 0, len(conditionEdges)+len(ids))
	for _, e := range conditionEdges {
		sources = append(sources, train.ResidentLeaseEdge{Source: b.handoffLease, Output: e.Output, Input: e.Input})
	}
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	commands := make([]train.Command, len(t.Commands))
	mapping := map[int]int{}
	for i, id := range ids {
		if _, duplicate := mapping[id]; duplicate {
			return nil, fmt.Errorf("prediction duplicate edge")
		}
		placeholder := len(t.Tensors)
		t.Tensors = append(t.Tensors, train.TensorSpec{Shape: append([]int(nil), t.Tensors[id].Shape...), Values: values[i]})
		mapping[id] = placeholder
		sources = append(sources, train.ResidentLeaseEdge{Source: b.predictionLease, Output: b.predictionRoots[i], Input: placeholder})
	}
	for i, c := range t.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		for j, id := range c.Inputs {
			if replacement, ok := mapping[id]; ok {
				c.Inputs[j] = replacement
			}
		}
		commands[i] = c
	}
	t.Commands = commands
	if err := train.ValidateResidentTapeBindings(t, bindings, b.MaxTapeBytes); err != nil {
		return nil, err
	}
	return b.predictionLease.RunSourcesTape(b.ctx, t, bindings, nil, outputs, sources, true, scale)
}
func (b *ResidentTrainingBackend) closePredictionHandoff() error {
	if b == nil || b.predictionLease == nil {
		return nil
	}
	if err := b.predictionLease.Close(); err != nil {
		return b.fail(err)
	}
	b.predictionLease = nil
	b.predictionValues = nil
	b.predictionRoots = nil
	b.predictionKind = ""
	b.predictionRow = -1
	return nil
}
