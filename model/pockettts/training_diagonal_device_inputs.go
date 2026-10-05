package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private retained-root contract. Shape metadata never asserts numerical
// admission; the exact sources are first checked in a read-only device tape.
// All owners must belong to Root's live lease tree. The experimental
// resident-devicehandoff coordinator uses this contract with host guards intact.
type diagonalDeviceSources struct {
	Root                                  *train.ResidentTapeLease
	Row                                   int
	Condition, Input, Desired, Prediction diagonalDeviceRoot
	provenance                            *diagonalDeviceProvenance
}
type diagonalDeviceProvenance struct {
	root                         *train.ResidentTapeLease
	model                        *FlowHeadCPU
	flow                         *FlowHeadCPU // frozen graph metadata; weights bind the same resident state
	row, conditionSize, channels int
	time                         float32
	roots                        [4]diagonalDeviceRoot
	consumed                     bool // one attempt; duplicate copies cannot accumulate twice
}
type diagonalDeviceRoot struct {
	Owner  *train.ResidentTapeLease
	Output int
}
type diagonalDeviceIDs struct{ Condition, Input, Desired, Prediction int }
type diagonalDeviceTape struct {
	tape                    train.Tape
	bindings                map[string]train.ResidentTapeBinding
	ids                     diagonalDeviceIDs
	conditionSize, channels int
}

func (ids diagonalDeviceIDs) edges(s diagonalDeviceSources) []train.ResidentLeaseEdge {
	return []train.ResidentLeaseEdge{{Source: s.Condition.Owner, Output: s.Condition.Output, Input: ids.Condition}, {Source: s.Input.Owner, Output: s.Input.Output, Input: ids.Input}, {Source: s.Desired.Owner, Output: s.Desired.Output, Input: ids.Desired}, {Source: s.Prediction.Owner, Output: s.Prediction.Output, Input: ids.Prediction}}
}
func validateDiagonalDeviceMetadata(m *FlowHeadCPU, conditionSize, channels int, time, logvar, scale float32, maxBytes int64) error {
	if m == nil || len(m.Time) != 2 || channels != m.Input.In || channels != m.Final.Linear.Out {
		return fmt.Errorf("device diagonal topology/extent")
	}
	if err := validateTrainableFlowHeadSizes(m, conditionSize, 2, channels, channels); err != nil {
		return err
	}
	if !isFinite(time) || time < 0 || time > 1 || !isFinite(scale) || scale < 0 || scale > 1 {
		return fmt.Errorf("device diagonal time/scale")
	}
	if _, err := train.LSDPredictionCutoff(channels, logvar); err != nil {
		return err
	}
	if maxBytes <= 0 || int64(4*(3*channels+4)) > maxBytes {
		return fmt.Errorf("device diagonal loss byte ceiling")
	}
	return nil
}
func validateDiagonalDeviceIDs(t train.Tape, ids diagonalDeviceIDs, conditionSize, channels int, maxBytes int64) error {
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	seen := map[int]bool{}
	for _, p := range []struct{ id, n int }{{ids.Condition, conditionSize}, {ids.Input, channels}, {ids.Desired, channels}, {ids.Prediction, channels}} {
		if p.id < 0 || p.id >= len(t.Tensors) || seen[p.id] {
			return fmt.Errorf("device diagonal source ID/duplicate")
		}
		spec := t.Tensors[p.id]
		if !spec.External || len(spec.Shape) != 1 || spec.Shape[0] != p.n {
			return fmt.Errorf("device diagonal source extent/declaration")
		}
		seen[p.id] = true
	}
	return train.ValidateExternalTapeBindings(t, seen)
}

// Admission has no named gradients: LSD cutoff flag checks exact prediction and
// desired roots, finite witness checks condition/input too. Sources remain in
// full admission even if pruning would discard them. Runtime reads two scalars.
func buildDiagonalDeviceAdmission(conditionSize, channels int, logvar float32, maxBytes int64) (train.Tape, diagonalDeviceIDs, error) {
	cutoff, err := train.LSDPredictionCutoff(channels, logvar)
	if err != nil {
		return train.Tape{}, diagonalDeviceIDs{}, err
	}
	g := &flowTapeGraph{}
	ids := diagonalDeviceIDs{g.external(conditionSize), g.external(channels), g.external(channels), g.external(channels)}
	flag := g.command("lsd-loss", []int{ids.Prediction, ids.Desired, g.constant([]float32{cutoff})}, []int{1}, 0, 2, 0, logvar, 0)
	finiteGraph := &transformerTapeGraph{tape: g.tape, zeros: map[int]bool{}, constants: map[int][]float32{}}
	witness := appendTrainingFiniteWitness(finiteGraph, []int{ids.Condition, ids.Input, ids.Desired, ids.Prediction})
	for id := range finiteGraph.zeros {
		n, _ := finiteGraph.tape.Tensors[id].Elements()
		finiteGraph.tape.Tensors[id].Values = make([]float32, n)
	}
	g.tape = finiteGraph.tape
	g.tape.Outputs = []int{flag, witness}
	if err := validateDiagonalDeviceIDs(g.tape, ids, conditionSize, channels, maxBytes); err != nil {
		return train.Tape{}, diagonalDeviceIDs{}, err
	}
	return g.tape, ids, nil
}
func (m *FlowHeadCPU) buildDiagonalDeviceTrainingTape(conditionSize, channels int, time, logvar, scale float32, maxBytes int64) (diagonalDeviceTape, error) {
	if err := validateDiagonalDeviceMetadata(m, conditionSize, channels, time, logvar, scale, maxBytes); err != nil {
		return diagonalDeviceTape{}, err
	}
	sources := flowDeviceTapeSources{sizes: flowDeviceInputSizes{conditionSize, channels, channels}}
	var g *flowTapeGraph
	var desired, prediction, stats, scaledSeed int
	p, err := m.buildFlowDualTrainingTapeSources(nil, []float32{time, time}, nil, -1, nil, nil, maxBytes, true, true, func(graph *flowTapeGraph, out graphDual) (int, int) {
		g = graph
		desired, prediction = g.external(channels), g.external(channels)
		lv := g.constant([]float32{logvar})
		stats = g.command("lsd-loss", []int{prediction, desired, lv}, []int{3}, 0, 0, 0, 0, 0)
		seed := g.command("lsd-loss", []int{prediction, desired, lv}, []int{channels}, 0, 1, 0, 0, 0)
		scaledSeed = g.command("vector", []int{g.zero(channels), seed}, []int{channels}, 0, 0, 0, scale, 0)
		return scaledSeed, g.zero(channels)
	}, &sources)
	if err != nil {
		return diagonalDeviceTape{}, err
	}
	if len(p.Tape.Outputs) < 4 || p.inputSize != channels || p.conditionSize != conditionSize || p.outputSize != channels {
		return diagonalDeviceTape{}, fmt.Errorf("device diagonal flow boundary contract")
	}
	for _, v := range []struct{ index, n int }{{2, conditionSize}, {3, channels}} {
		id := p.Tape.Outputs[v.index]
		if id < 0 || id >= len(p.Tape.Tensors) {
			return diagonalDeviceTape{}, fmt.Errorf("device diagonal adjoint root ID")
		}
		n, e := p.Tape.Tensors[id].Elements()
		if e != nil || n != v.n {
			return diagonalDeviceTape{}, fmt.Errorf("device diagonal adjoint extent")
		}
	}
	// Preserve time*dX - scaled prediction seed; raw loss stats stay unscaled.
	scaledInput := g.command("vector", []int{g.zero(channels), p.Tape.Outputs[3]}, []int{channels}, 0, 0, 0, time, 0)
	dtarget := g.command("vector", []int{scaledInput, scaledSeed}, []int{channels}, 0, 0, 0, -1, 0)
	g.tape.Outputs = []int{p.Tape.Outputs[2], dtarget, stats}
	for _, name := range p.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, p.ResidentBindings[name].Gradient)
	}
	ids := diagonalDeviceIDs{sources.ids.Condition, sources.ids.Input, desired, prediction}
	if err := validateDiagonalDeviceIDs(g.tape, ids, conditionSize, channels, maxBytes); err != nil {
		return diagonalDeviceTape{}, err
	}
	if err := train.ValidateResidentTapeBindings(g.tape, p.ResidentBindings, maxBytes); err != nil {
		return diagonalDeviceTape{}, err
	}
	return diagonalDeviceTape{g.tape, p.ResidentBindings, ids, conditionSize, channels}, nil
}

// runDiagonalDeviceTraining is private to the experimental resident-devicehandoff
// coordinator; its opt-in CLI scope retains host downloads and overflow guards.
// It checks numerical source admission before constructing/running backward,
// keeps all producers alive through accumulation, and retains failures rather
// than publishing a result or releasing uncertain owners. Caller serialises.
func runDiagonalDeviceTraining(ctx context.Context, m *FlowHeadCPU, s diagonalDeviceSources, row, conditionSize, channels int, time, logvar, scale float32, maxBytes int64) (result DiagonalTrainingResult, err error) {
	if s.Root == nil {
		return result, fmt.Errorf("device diagonal nil root")
	}
	defer func() {
		if err != nil {
			err = s.Root.RetainFailure(err)
			result = DiagonalTrainingResult{}
		}
	}()
	p := s.provenance
	if ctx == nil || s.Row < 0 || s.Row != row || p == nil || p.root != s.Root || p.model != m || p.flow == nil || p.row != row || p.conditionSize != conditionSize || p.channels != channels || p.time != time || p.roots != [4]diagonalDeviceRoot{s.Condition, s.Input, s.Desired, s.Prediction} {
		return result, fmt.Errorf("device diagonal context/owner/row/root provenance")
	}
	if p.consumed {
		return result, fmt.Errorf("device diagonal already consumed")
	}
	p.consumed = true
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if err = validateDiagonalDeviceMetadata(p.flow, conditionSize, channels, time, logvar, scale, maxBytes); err != nil {
		return result, err
	}
	admission, ids, e := buildDiagonalDeviceAdmission(conditionSize, channels, logvar, maxBytes)
	if e != nil {
		return result, e
	}
	flags, e := s.Root.RunSourcesTape(ctx, admission, nil, nil, admission.Outputs, ids.edges(s), false, 1)
	if e != nil {
		return result, e
	}
	if len(flags) != 2 || len(flags[0]) != 1 || len(flags[1]) != 1 || !finiteF32(flags[0]) || !finiteF32(flags[1]) || flags[0][0] != 0 || flags[1][0] != 0 || f32SlicesOverlap(flags[0], flags[1]) {
		return result, fmt.Errorf("device diagonal scalar admission")
	}
	backward, e := p.flow.buildDiagonalDeviceTrainingTape(conditionSize, channels, time, logvar, scale, maxBytes)
	if e != nil {
		return result, e
	}
	values, e := s.Root.RunSourcesTape(ctx, backward.tape, backward.bindings, nil, backward.tape.Outputs[:3], backward.ids.edges(s), true, 1)
	if e != nil {
		return result, e
	}
	if len(values) != 3 || len(values[0]) != conditionSize || len(values[1]) != channels || len(values[2]) != 3 {
		return result, fmt.Errorf("device diagonal result shape")
	}
	for i, v := range values {
		if !finiteF32(v) {
			return result, fmt.Errorf("device diagonal result nonfinite")
		}
		for _, other := range values[:i] {
			if f32SlicesOverlap(v, other) {
				return result, fmt.Errorf("device diagonal result alias")
			}
		}
	}
	if values[2][0] < 0 {
		return result, fmt.Errorf("device diagonal negative square")
	}
	// The exact forward child minted by begin remains live through all reads
	// and accumulation. Close it before publishing this row; parent stays live.
	if err = s.Input.Owner.Close(); err != nil {
		return result, err
	}
	return DiagonalTrainingResult{DCondition: values[0], DTarget: values[1], RawSquare: float64(values[2][0]), Loss: float64(values[2][1]), DLogVariance: values[2][2]}, nil
}
