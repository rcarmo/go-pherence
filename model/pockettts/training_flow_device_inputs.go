package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private graph-only contract: extents and explicit read-only source IDs. These
// carry no numerical approval or device-owner identity. Execution must provide
// all four bindings through a live retained lease and preserve finite/overflow
// admission before any gradient accumulation. No production caller uses it yet.
type flowDeviceInputSizes struct {
	Condition, Input, Output int
}
type flowDeviceInputIDs struct {
	Condition, Input, ValueSeed, TangentSeed int
}
type flowDeviceTapeSources struct {
	sizes flowDeviceInputSizes
	ids   flowDeviceInputIDs
}
type flowDeviceTrainingTape struct {
	flow   FlowDualTrainingTape
	inputs flowDeviceInputIDs
}

func (g *flowTapeGraph) external(size int) int {
	id := g.tensor([]int{size}, nil)
	g.tape.Tensors[id].External = true
	return id
}

// buildFlowDeviceInputsTrainingTape assembles the same reverse-over-forward
// graph as the device-time host-input builder, with only condition/input/value
// seed/tangent seed represented as shape-only external sources. Literal-zero
// input tangents remain genuine constants; times/frequencies stay host-admitted.
// This synchronous private builder borrows parameters read-only for immediate
// resident rebinding. It does not remove forward graphs, guards or downloads.
func (m *FlowHeadCPU) buildFlowDeviceInputsTrainingTape(sizes flowDeviceInputSizes, times []float32, timeIndex int, maxBytes int64) (flowDeviceTrainingTape, error) {
	sources := flowDeviceTapeSources{sizes: sizes}
	flow, err := m.buildFlowDualTrainingTapeSources(nil, times, nil, timeIndex, nil, nil, maxBytes, true, true, nil, &sources)
	if err != nil {
		return flowDeviceTrainingTape{}, err
	}
	return flowDeviceTrainingTape{flow: flow, inputs: sources.ids}, nil
}

// Validate the complete graph and declared source IDs before pruning. Keep the
// method private: a caller can validate shape/provenance here without passing
// numerical admission or gaining authority to accumulate device gradients.
func (t flowDeviceTrainingTape) validate(maxBytes int64) error {
	if err := t.flow.Tape.Validate(maxBytes); err != nil {
		return err
	}
	// Generic read-only admission verifies all externals are exactly accounted
	// for. Duplicate/tampered source IDs leave a source unbound and are rejected.
	bindings := map[int]bool{}
	for _, source := range []struct{ id, size int }{{t.inputs.Condition, t.flow.conditionSize}, {t.inputs.Input, t.flow.inputSize}, {t.inputs.ValueSeed, t.flow.outputSize}, {t.inputs.TangentSeed, t.flow.outputSize}} {
		if source.id < 0 || source.id >= len(t.flow.Tape.Tensors) || bindings[source.id] {
			return fmt.Errorf("device flow source ID/duplicate")
		}
		spec := t.flow.Tape.Tensors[source.id]
		if !spec.External || len(spec.Shape) != 1 || spec.Shape[0] != source.size {
			return fmt.Errorf("device flow source shape/declaration")
		}
		bindings[source.id] = true
	}
	if t.inputs.Input != t.flow.inputValue {
		return fmt.Errorf("device flow input provenance")
	}
	return train.ValidateExternalTapeBindings(t.flow.Tape, bindings)
}
