package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private single-owner, caller-serialised retained forward provenance for the
// experimental resident-devicehandoff coordinator. The opt-in CLI scope retains
// host downloads and guards; this contract grants no native qualification.
type distillDeviceSources struct {
	root       *train.ResidentTapeLease
	condition  diagonalDeviceRoot
	forward    [6]diagonalDeviceRoot
	provenance *distillDeviceProvenance
}
type distillDeviceProvenance struct {
	root                         *train.ResidentTapeLease
	model, flow                  *FlowHeadCPU
	row, conditionSize, channels int
	s, t                         float32
	target, noise                []float32
	condition                    diagonalDeviceRoot
	forward                      [6]diagonalDeviceRoot
	consumed                     bool
}

func (ids distillDeviceIDs) edges(s distillDeviceSources) []train.ResidentLeaseEdge {
	edges := []train.ResidentLeaseEdge{{Source: s.condition.Owner, Output: s.condition.Output, Input: ids.Condition}}
	for i, r := range s.forward {
		edges = append(edges, train.ResidentLeaseEdge{Source: r.Owner, Output: r.Output, Input: ids.Forward[i]})
	}
	return edges
}
func beginDistillDeviceSources(ctx context.Context, root *train.ResidentTapeLease, m *FlowHeadCPU, row int, condition diagonalDeviceRoot, in DistillForwardInputs, maxBytes int64) (sources distillDeviceSources, err error) {
	if root == nil {
		return sources, fmt.Errorf("device distill nil root")
	}
	defer func() {
		if err != nil {
			err = root.RetainFailure(err)
			sources = distillDeviceSources{}
		}
	}()
	if ctx == nil || row < 0 || condition.Owner == nil {
		return sources, fmt.Errorf("device distill context/row/condition")
	}
	if err = ctx.Err(); err != nil {
		return sources, err
	}
	flow := freezeDiagonalDeviceFlow(m)
	forward, err := flow.buildDistillForwardTape(in, maxBytes, true)
	if err != nil {
		return sources, err
	}
	if err = validateDistillDeviceForwardRoots(forward, len(in.Target), maxBytes); err != nil {
		return sources, err
	}
	b, ok := forward.ResidentBindings["flow_lm.flow_net.cond_embed.weight"]
	if !ok {
		return sources, fmt.Errorf("device distill missing condition parameter")
	}
	writers := map[int]train.Command{}
	for _, c := range forward.Tape.Commands {
		writers[c.Output] = c
	}
	ids := []int{}
	seen := map[int]bool{}
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
			if !seen[id] {
				ids = append(ids, id)
				seen[id] = true
			}
		}
	}
	if len(ids) != 2 {
		return sources, fmt.Errorf("device distill condition source count")
	}
	tape, err := externalizeImmediateChild(forward.Tape, ids, maxBytes)
	if err != nil {
		return sources, err
	}
	// Both immutable condition sources represent the same admitted row. Read one
	// retained root and keep unused second source extent charged before pruning.
	for i, c := range tape.Commands {
		inputs := append([]int(nil), c.Inputs...)
		for j, id := range inputs {
			if id == ids[1] {
				inputs[j] = ids[0]
			}
		}
		tape.Commands[i].Inputs = inputs
	}
	tape.Tensors[ids[1]].External = false
	if err = train.ValidateResidentTapeBindings(tape, forward.ResidentBindings, maxBytes); err != nil {
		return sources, err
	}
	child, err := root.BeginChildTape(ctx, tape, forward.ResidentBindings, tape.Outputs[:6], []train.ResidentLeaseEdge{{Source: condition.Owner, Output: condition.Output, Input: ids[0]}})
	if err != nil {
		return sources, err
	}
	sources = distillDeviceSources{root: root, condition: condition}
	for i := range sources.forward {
		sources.forward[i] = diagonalDeviceRoot{child, tape.Outputs[i]}
	}
	sources.provenance = &distillDeviceProvenance{root: root, model: m, flow: flow, row: row, conditionSize: len(in.Condition), channels: len(in.Target), s: in.S, t: in.T, target: append([]float32(nil), in.Target...), noise: append([]float32(nil), in.Noise...), condition: condition, forward: sources.forward}
	return sources, nil
}
func validateDistillDeviceForwardRoots(p DistillForwardTape, n int, maxBytes int64) error {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return err
	}
	if len(p.Tape.Outputs) < 6 {
		return fmt.Errorf("device distill forward root count")
	}
	seen := map[int]bool{}
	for _, id := range p.Tape.Outputs[:6] {
		if id < 0 || id >= len(p.Tape.Tensors) || seen[id] {
			return fmt.Errorf("device distill forward root ID/alias")
		}
		seen[id] = true
		spec := p.Tape.Tensors[id]
		if len(spec.Shape) != 1 || spec.Shape[0] != n {
			return fmt.Errorf("device distill forward root extent")
		}
	}
	return nil
}
func buildDistillDeviceAdmission(conditionSize, n int, logvar float32, maxBytes int64) (train.Tape, distillDeviceIDs, error) {
	cutoff, err := train.LSDPredictionCutoff(n, logvar)
	if err != nil {
		return train.Tape{}, distillDeviceIDs{}, err
	}
	g := &flowTapeGraph{}
	ids := distillDeviceIDs{Condition: g.external(conditionSize)}
	for i := range ids.Forward {
		ids.Forward[i] = g.external(n)
	}
	flag := g.command("lsd-loss", []int{ids.Forward[5], ids.Forward[4], g.constant([]float32{cutoff})}, []int{1}, 0, 2, 0, logvar, 0)
	fg := &transformerTapeGraph{tape: g.tape, zeros: map[int]bool{}, constants: map[int][]float32{}}
	witness := appendTrainingFiniteWitness(fg, append([]int{ids.Condition}, ids.Forward[:]...))
	for id := range fg.zeros {
		extent, _ := fg.tape.Tensors[id].Elements()
		fg.tape.Tensors[id].Values = make([]float32, extent)
	}
	fg.tape.Outputs = []int{flag, witness}
	if err := validateDistillDeviceIDs(fg.tape, ids, conditionSize, n, maxBytes); err != nil {
		return train.Tape{}, distillDeviceIDs{}, err
	}
	return fg.tape, ids, nil
}
func runDistillDeviceTraining(ctx context.Context, m *FlowHeadCPU, s distillDeviceSources, row int, logvar, parameterScale float32, maxBytes int64) (result DistillTrainingResult, err error) {
	if s.root == nil {
		return result, fmt.Errorf("device distill nil root")
	}
	defer func() {
		if err != nil {
			err = s.root.RetainFailure(err)
			result = DistillTrainingResult{}
		}
	}()
	p := s.provenance
	if ctx == nil || p == nil || p.root != s.root || p.model != m || p.flow == nil || p.row != row || p.condition != s.condition || p.forward != s.forward {
		return result, fmt.Errorf("device distill provenance")
	}
	if p.consumed {
		return result, fmt.Errorf("device distill already consumed")
	}
	p.consumed = true
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if !isFinite(parameterScale) || parameterScale < 0 {
		return result, fmt.Errorf("device distill parameter scale")
	}
	if err = validateDistillDeviceMetadata(p.flow, p.conditionSize, p.channels, p.s, p.t, logvar, maxBytes); err != nil {
		return result, err
	}
	admission, ids, e := buildDistillDeviceAdmission(p.conditionSize, p.channels, logvar, maxBytes)
	if e != nil {
		return result, e
	}
	flags, e := s.root.RunSourcesTape(ctx, admission, nil, nil, admission.Outputs, ids.edges(s), false, 1)
	if e != nil {
		return result, e
	}
	if len(flags) != 2 || len(flags[0]) != 1 || len(flags[1]) != 1 || !finiteF32(flags[0]) || !finiteF32(flags[1]) || flags[0][0] != 0 || flags[1][0] != 0 || f32SlicesOverlap(flags[0], flags[1]) {
		return result, fmt.Errorf("device distill scalar admission")
	}
	backward, e := p.flow.buildScaledDistillDeviceTrainingTape(p.conditionSize, p.channels, p.s, p.t, logvar, p.target, p.noise, maxBytes, parameterScale)
	if e != nil {
		return result, e
	}
	values, e := s.root.RunSourcesTape(ctx, backward.tape, backward.bindings, nil, backward.tape.Outputs[:5], backward.ids.edges(s), true, 1)
	if e != nil {
		return result, e
	}
	if len(values) != 5 {
		return result, fmt.Errorf("device distill result count")
	}
	for i, v := range values {
		want := []int{p.conditionSize, p.channels, p.channels, 2, 3}[i]
		if len(v) != want || !finiteF32(v) {
			return result, fmt.Errorf("device distill result shape/nonfinite")
		}
		for _, other := range values[:i] {
			if f32SlicesOverlap(v, other) {
				return result, fmt.Errorf("device distill result alias")
			}
		}
	}
	if values[4][0] < 0 {
		return result, fmt.Errorf("device distill negative square")
	}
	if err = s.forward[0].Owner.Close(); err != nil {
		return result, err
	}
	return DistillTrainingResult{PrimaryTrainingResult: PrimaryTrainingResult{values[0], values[1], values[2], values[3]}, RawSquare: float64(values[4][0]), Loss: float64(values[4][1]), DLogVariance: values[4][2]}, nil
}
