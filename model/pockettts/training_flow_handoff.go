package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (b *ResidentTrainingBackend) flowHandoffEnabled() bool {
	return b != nil && (b.eosAdmissionEnabled() || b.Scope == "ptx-resident-flowhandoff-hybrid-v1" || b.Scope == "vulkan-resident-flowhandoff-hybrid-v1")
}

// bindFlowHandoffCondition preserves the complete admitted row graph and IDs.
// It replaces only reads of the explicit condition constants, with a row slice
// of the pinned producer root. Unused old constants stay in full admission.
func (b *ResidentTrainingBackend) bindFlowHandoffCondition(t train.Tape, bindings map[string]train.ResidentTapeBinding, condition []float32) (train.Tape, []train.TapeHandoffEdge, error) {
	if b.handoffLease == nil {
		return t, nil, nil
	}
	if !b.flowHandoffEnabled() || b.handoffRow < 0 || b.handoffWidth <= 0 || b.handoffRow >= b.handoffFrames || b.handoffWidth > len(b.handoffHidden) || b.handoffFrames != len(b.handoffHidden)/b.handoffWidth || len(b.handoffHidden)%b.handoffWidth != 0 || len(condition) != b.handoffWidth {
		return train.Tape{}, nil, fmt.Errorf("flow handoff row/owner")
	}
	expected := b.handoffHidden[b.handoffRow*b.handoffWidth : (b.handoffRow+1)*b.handoffWidth]
	if len(expected) != len(condition) || &expected[0] != &condition[0] {
		return train.Tape{}, nil, fmt.Errorf("flow handoff condition provenance")
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, b.MaxTapeBytes); err != nil {
		return train.Tape{}, nil, err
	}
	parameter, ok := bindings["flow_lm.flow_net.cond_embed.weight"]
	if !ok {
		return train.Tape{}, nil, fmt.Errorf("flow handoff condition parameter")
	}
	aliases := map[int]int{}
	for _, c := range t.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = c.Inputs[0]
		}
	}
	ids := map[int]bool{}
	for _, c := range t.Commands {
		if c.Op == "linear-forward" && c.Mode == 0 && len(c.Inputs) == 4 && c.Inputs[2] == parameter.Parameter {
			id := c.Inputs[0]
			for {
				source, ok := aliases[id]
				if !ok {
					break
				}
				id = source
			}
			ids[id] = true
		}
	}
	if len(ids) == 0 {
		return train.Tape{}, nil, fmt.Errorf("flow handoff missing condition edge")
	}
	for id := range ids {
		for _, binding := range bindings {
			if binding.Parameter == id {
				return train.Tape{}, nil, fmt.Errorf("flow handoff condition/parameter alias")
			}
		}
		n, _ := t.Tensors[id].Elements()
		if n != len(condition) || t.Tensors[id].Values == nil {
			return train.Tape{}, nil, fmt.Errorf("flow handoff constant extent id=%d n=%d want=%d constant=%t", id, n, len(condition), t.Tensors[id].Values != nil)
		}
		for i, value := range t.Tensors[id].Values {
			if value != condition[i] {
				return train.Tape{}, nil, fmt.Errorf("flow handoff condition value provenance")
			}
		}
	}
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	commands := make([]train.Command, 0, len(t.Commands)+1)
	root := len(t.Tensors)
	unused, row := root+1, root+2
	t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{len(b.handoffHidden)}, Values: b.handoffHidden}, train.TensorSpec{Shape: []int{1}, Values: []float32{0}}, train.TensorSpec{Shape: []int{len(condition)}})
	commands = append(commands, train.Command{Op: "vector", Inputs: []int{root, unused}, Output: row, Mode: 1, Offset: b.handoffRow * b.handoffWidth})
	for _, c := range t.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		for i, id := range c.Inputs {
			if ids[id] {
				c.Inputs[i] = row
			}
		}
		commands = append(commands, c)
	}
	t.Commands = commands
	if err := train.ValidateResidentTapeBindings(t, bindings, b.MaxTapeBytes); err != nil {
		return train.Tape{}, nil, err
	}
	return t, []train.TapeHandoffEdge{{Output: b.handoffRoot, Input: root}}, nil
}
func (b *ResidentTrainingBackend) runFlowHandoffTape(t train.Tape, bindings map[string]train.ResidentTapeBinding, outputs []int, condition []float32) ([][]float32, error) {
	var err error
	var edges []train.TapeHandoffEdge
	t, edges, err = b.bindFlowHandoffCondition(t, bindings, condition)
	if err != nil {
		return nil, err
	}
	if b.handoffLease == nil {
		return b.runResidentTape(t, bindings, outputs, false, 1)
	}
	return b.handoffLease.RunTape(b.ctx, t, bindings, outputs, edges, false, 1)
}
func (b *ResidentTrainingBackend) closeFlowHandoff() error {
	if b == nil || b.handoffLease == nil {
		return nil
	}
	if b.predictionLease != nil || b.deviceCondition != nil || b.deviceDiagonal != nil || b.deviceDistill != nil {
		return b.fail(fmt.Errorf("prediction child still live"))
	}
	if err := b.handoffLease.Close(); err != nil {
		return b.fail(err)
	}
	b.handoffLease = nil
	b.handoffHidden = nil
	b.handoffRow = -1
	return nil
}
func flowBackendForHandoff(flow *FlowHeadCPU) (*ResidentTrainingBackend, bool) {
	if flow == nil {
		return nil, false
	}
	b, ok := flow.TrainingBackend.(*ResidentTrainingBackend)
	return b, ok && b.flowHandoffEnabled()
}
