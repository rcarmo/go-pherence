package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// beginDiagonalDeviceSources records provenance only for the exact private
// forward graph it constructs. Caller passes an already pinned row condition.
// Raw metadata cannot mint a numerical admission; run checks scalar guards.
// The experimental resident-devicehandoff coordinator calls this helper and
// preserves original host forward input admission and downloads. Run closes
// the child before row publication; native qualification is still required.
func beginDiagonalDeviceSources(ctx context.Context, root *train.ResidentTapeLease, m *FlowHeadCPU, row int, condition diagonalDeviceRoot, in DiagonalForwardInputs, maxBytes int64) (sources diagonalDeviceSources, err error) {
	if root == nil {
		return sources, fmt.Errorf("device diagonal nil parent")
	}
	defer func() {
		if err != nil {
			err = root.RetainFailure(err)
			sources = diagonalDeviceSources{}
		}
	}()
	if ctx == nil || row < 0 || condition.Owner == nil {
		return sources, fmt.Errorf("device diagonal row/condition owner")
	}
	flow := freezeDiagonalDeviceFlow(m)
	forward, err := flow.buildDiagonalForwardTape(in, maxBytes, true)
	if err != nil {
		return sources, err
	}
	if err = validateDiagonalDeviceForwardRoots(forward, len(in.Target), maxBytes); err != nil {
		return sources, err
	}
	ids := []int{}
	// Exact flow condition source, following only reshape aliases. No numerical
	// placeholder replaces the caller's admitted host condition; remove values
	// only after complete forward graph/binding admission.
	b, ok := forward.ResidentBindings["flow_lm.flow_net.cond_embed.weight"]
	if !ok {
		return sources, fmt.Errorf("device diagonal missing condition parameter")
	}
	writers := map[int]train.Command{}
	for _, c := range forward.Tape.Commands {
		writers[c.Output] = c
	}
	for _, c := range forward.Tape.Commands {
		if c.Op == "linear-forward" && len(c.Inputs) == 4 && c.Inputs[2] == b.Parameter {
			id := c.Inputs[0]
			for {
				w, ok := writers[id]
				if !ok || w.Op != "reshape" || len(w.Inputs) != 1 {
					break
				}
				id = w.Inputs[0]
			}
			if len(ids) == 0 {
				ids = append(ids, id)
			} else if ids[0] != id {
				return sources, fmt.Errorf("device diagonal ambiguous condition")
			}
		}
	}
	if len(ids) != 1 {
		return sources, fmt.Errorf("device diagonal condition source")
	}
	tape, err := externalizeImmediateChild(forward.Tape, ids, maxBytes)
	if err != nil {
		return sources, err
	}
	child, err := root.BeginChildTape(ctx, tape, forward.ResidentBindings, tape.Outputs[:3], []train.ResidentLeaseEdge{{Source: condition.Owner, Output: condition.Output, Input: ids[0]}})
	if err != nil {
		return sources, err
	}
	sources = diagonalDeviceSources{Root: root, Row: row, Condition: condition, Input: diagonalDeviceRoot{child, tape.Outputs[0]}, Desired: diagonalDeviceRoot{child, tape.Outputs[1]}, Prediction: diagonalDeviceRoot{child, tape.Outputs[2]}}
	sources.provenance = &diagonalDeviceProvenance{root: root, model: m, flow: flow, row: row, conditionSize: len(in.Condition), channels: len(in.Target), time: in.Time, roots: [4]diagonalDeviceRoot{sources.Condition, sources.Input, sources.Desired, sources.Prediction}}
	return sources, nil
}

// Capture arithmetic metadata, never numerical stand-ins for device inputs.
// Parameter values are only shape/finite-admitted host inventory and rebound
// by name from the unchanged resident state in forward and backward. Copying
// frequencies and value-owned topology prevents caller metadata edits from
// changing recomputation while the producer lease is pinned.
func freezeDiagonalDeviceFlow(m *FlowHeadCPU) *FlowHeadCPU {
	if m == nil {
		return nil
	}
	flow := *m
	flow.Time = append([]TimestepMLP(nil), m.Time...)
	for i := range flow.Time {
		flow.Time[i].Frequencies = append([]float32(nil), m.Time[i].Frequencies...)
	}
	flow.Blocks = append([]AdaLNResidual(nil), m.Blocks...)
	return &flow
}

// Fixed private forward contract: distinct flat input/desired/prediction roots,
// each with channel extent. Builder semantic order is exercised independently
// in composition/VJP tests; do not infer it from map order or matching extents.
func validateDiagonalDeviceForwardRoots(p DiagonalForwardTape, channels int, maxBytes int64) error {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return err
	}
	if len(p.Tape.Outputs) < 3 {
		return fmt.Errorf("device diagonal forward root count")
	}
	seen := map[int]bool{}
	for _, id := range p.Tape.Outputs[:3] {
		if id < 0 || id >= len(p.Tape.Tensors) || seen[id] {
			return fmt.Errorf("device diagonal forward root ID/alias")
		}
		seen[id] = true
		s := p.Tape.Tensors[id]
		if len(s.Shape) != 1 || s.Shape[0] != channels {
			return fmt.Errorf("device diagonal forward root extent")
		}
	}
	return nil
}
