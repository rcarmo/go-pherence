package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private graph-only descriptors for the experimental resident-devicehandoff
// coordinator. Runtime supplies exact retained sources only after scalar
// admission, without host numerical placeholders; host guards remain intact.
type distillDeviceIDs struct {
	Condition int
	Forward   [6]int // primary input, velocity, tangent, endpoint input, desired, prediction
}
type distillDeviceTape struct {
	tape     train.Tape
	bindings map[string]train.ResidentTapeBinding
	ids      distillDeviceIDs
}

func validateDistillDeviceMetadata(m *FlowHeadCPU, conditionSize, n int, s, t, logvar float32, maxBytes int64) error {
	if !isFinite(s) || !isFinite(t) || s < 0 || s > t || t > 1 {
		return fmt.Errorf("device distill times")
	}
	return validateDiagonalDeviceMetadata(m, conditionSize, n, t, logvar, 1, maxBytes)
}
func (m *FlowHeadCPU) buildEndpointDeviceTape(conditionSize, n int, t, logvar, delta float32, maxBytes int64) (EndpointTrainingTape, int, error) {
	sources := flowDeviceTapeSources{sizes: flowDeviceInputSizes{conditionSize, n, n}}
	var g *flowTapeGraph
	var desired, prediction, stats, dr int
	p, err := m.buildFlowDualTrainingTapeSources(nil, []float32{t, t}, nil, -1, nil, nil, maxBytes, true, true, func(graph *flowTapeGraph, out graphDual) (int, int) {
		g = graph
		desired, prediction = g.external(n), g.external(n)
		lv := g.constant([]float32{logvar})
		stats = g.command("lsd-loss", []int{desired, prediction, lv}, []int{3}, 0, 0, 0, 0, 0)
		dr = g.command("lsd-loss", []int{desired, prediction, lv}, []int{n}, 0, 1, 0, 0, 0)
		return g.command("vector", []int{g.zero(n), dr}, []int{n}, 0, 0, 0, -1, 0), g.zero(n)
	}, &sources)
	if err != nil {
		return EndpointTrainingTape{}, 0, err
	}
	if len(p.Tape.Outputs) < 4 {
		return EndpointTrainingTape{}, 0, fmt.Errorf("device endpoint flow boundary")
	}
	dx := p.Tape.Outputs[3]
	gv := g.command("vector", []int{dr, dx}, []int{n}, 0, 0, 0, delta, 0)
	gt := g.command("vector", []int{g.zero(n), dr}, []int{n}, 0, 0, 0, delta, 0)
	dot := func(a, b int) int {
		return g.reshape(g.command("linear-forward", []int{g.reshape(a, 1, n), g.reshape(g.zero(n), 1, n), g.reshape(b, 1, n), g.zero(1)}, []int{1, 1}, 0, 0, 0, 0, 0), 1)
	}
	tangent, velocity := g.external(n), g.external(n)
	dd := g.command("vector", []int{dot(dr, tangent), dot(dx, velocity)}, []int{1}, 0, 0, 0, 1, 0)
	g.tape.Outputs = []int{dx, gv, gt, stats, dd}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return EndpointTrainingTape{}, 0, err
	}
	return EndpointTrainingTape{Tape: g.tape, ResidentBindings: p.ResidentBindings, inputID: p.inputValue, desiredID: desired, predictionID: prediction, velocityID: velocity, tangentID: tangent}, sources.ids.Condition, nil
}
func (m *FlowHeadCPU) buildPrimaryDeviceTape(conditionSize, n int, s, t float32, target, noise []float32, maxBytes int64) (PrimaryTrainingTape, int, error) {
	if len(target) != n || len(noise) != n || !finiteF32(target) || !finiteF32(noise) {
		return PrimaryTrainingTape{}, 0, fmt.Errorf("device primary target/noise")
	}
	sources := flowDeviceTapeSources{sizes: flowDeviceInputSizes{conditionSize, n, n}}
	var g *flowTapeGraph
	var seedIDs [4]int
	p, err := m.buildFlowDualTrainingTapeSources(nil, []float32{s, t}, nil, 1, nil, nil, maxBytes, true, true, func(graph *flowTapeGraph, out graphDual) (int, int) {
		g = graph
		seedIDs[0], seedIDs[1] = g.external(n), g.external(n)
		return seedIDs[0], seedIDs[1]
	}, &sources)
	if err != nil {
		return PrimaryTrainingTape{}, 0, err
	}
	if len(p.Tape.Outputs) < 6 {
		return PrimaryTrainingTape{}, 0, fmt.Errorf("device primary flow boundary")
	}
	add := func(a, b int, scale float32) int {
		return g.command("vector", []int{a, b}, []int{g.elements(a)}, 0, 0, 0, scale, 0)
	}
	scale := func(a int, v float32) int { return add(g.zero(g.elements(a)), a, v) }
	dot := func(a, b int) int {
		return g.reshape(g.command("linear-forward", []int{g.reshape(a, 1, n), g.reshape(g.zero(n), 1, n), g.reshape(b, 1, n), g.zero(1)}, []int{1, 1}, 0, 0, 0, 0, 0), 1)
	}
	seedIDs[2] = g.external(n)
	dx := add(p.Tape.Outputs[3], seedIDs[2], 1)
	dtarg, dnoise := scale(dx, s), scale(dx, 1-s)
	difference := add(g.constant(target), g.constant(noise), -1)
	seedIDs[3] = g.external(1)
	ds := add(add(dot(dx, difference), p.Tape.Outputs[4], 1), seedIDs[3], -1)
	dt := add(p.Tape.Outputs[5], seedIDs[3], 1)
	pad := func(a, offset int) int { return g.command("vector", []int{a, g.zero(1)}, []int{2}, 0, 2, offset, 0, 0) }
	times := add(pad(ds, 0), pad(dt, 1), 1)
	g.tape.Outputs = []int{p.Tape.Outputs[2], dtarg, dnoise, times}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return PrimaryTrainingTape{}, 0, err
	}
	return PrimaryTrainingTape{Tape: g.tape, ResidentBindings: p.ResidentBindings, seedIDs: seedIDs, inputID: p.inputValue}, sources.ids.Condition, nil
}
func (m *FlowHeadCPU) buildDistillDeviceTrainingTape(conditionSize, n int, s, t, logvar float32, target, noise []float32, maxBytes int64) (distillDeviceTape, error) {
	return m.buildScaledDistillDeviceTrainingTape(conditionSize, n, s, t, logvar, target, noise, maxBytes, 1)
}
func (m *FlowHeadCPU) buildScaledDistillDeviceTrainingTape(conditionSize, n int, s, t, logvar float32, target, noise []float32, maxBytes int64, scale float32) (distillDeviceTape, error) {
	if err := validateDistillDeviceMetadata(m, conditionSize, n, s, t, logvar, maxBytes); err != nil {
		return distillDeviceTape{}, err
	}
	endpoint, ec, err := m.buildEndpointDeviceTape(conditionSize, n, t, logvar, t-s, maxBytes)
	if err != nil {
		return distillDeviceTape{}, err
	}
	primary, pc, err := m.buildPrimaryDeviceTape(conditionSize, n, s, t, target, noise, maxBytes)
	if err != nil {
		return distillDeviceTape{}, err
	}
	composed, err := composeScaledDistillTrainingTapes(endpoint, primary, maxBytes, scale)
	if err != nil {
		return distillDeviceTape{}, err
	}
	// Scaled endpoint seed tensors precede the primary source region.
	offset := len(composed.Tape.Tensors) - len(primary.Tape.Tensors)
	// The component composer remaps the four explicit seeds to endpoint outputs.
	// Their old shape-only declarations now describe unused storage, not external
	// sources. Keep their logical extent charged, without inventing host values.
	for _, id := range primary.seedIDs {
		composed.Tape.Tensors[offset+id].External = false
	}
	// One condition root feeds both graphs. Eliminate the second external read
	// identity, retaining dead extent in complete-source admission.
	for i, c := range composed.Tape.Commands {
		if c.Output == offset+pc {
			return distillDeviceTape{}, fmt.Errorf("device distill condition write")
		}
		inputs := append([]int(nil), c.Inputs...)
		for j, id := range inputs {
			if id == offset+pc {
				inputs[j] = ec
			}
		}
		composed.Tape.Commands[i].Inputs = inputs
	}
	composed.Tape.Tensors[offset+pc].External = false
	ids := distillDeviceIDs{ec, composed.forwardIDs}
	if err := validateDistillDeviceIDs(composed.Tape, ids, conditionSize, n, maxBytes); err != nil {
		return distillDeviceTape{}, err
	}
	if err := train.ValidateResidentTapeBindings(composed.Tape, composed.ResidentBindings, maxBytes); err != nil {
		return distillDeviceTape{}, err
	}
	return distillDeviceTape{composed.Tape, composed.ResidentBindings, ids}, nil
}
func validateDistillDeviceIDs(t train.Tape, ids distillDeviceIDs, conditionSize, n int, maxBytes int64) error {
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	seen := map[int]bool{}
	all := append([]int{ids.Condition}, ids.Forward[:]...)
	for i, id := range all {
		extent := n
		if i == 0 {
			extent = conditionSize
		}
		if id < 0 || id >= len(t.Tensors) || seen[id] {
			return fmt.Errorf("device distill source ID/duplicate")
		}
		spec := t.Tensors[id]
		if !spec.External || len(spec.Shape) != 1 || spec.Shape[0] != extent {
			return fmt.Errorf("device distill source extent/declaration")
		}
		seen[id] = true
	}
	return train.ValidateExternalTapeBindings(t, seen)
}
