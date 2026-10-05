package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

type StepBackboneTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}
type stepBackboneTrainingResult struct {
	Reduction StepReductionResult
	Inputs    FlowLMTrainingInputGradients
}
type trainingStepBackboneBackend interface {
	stepBackboneEnabled() bool
	evaluateStepBackbone(*FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, FlowLMTrainingBatch, []float32, StepReductionInputs, TrainingStepConfig, WeightSampleInputs) (stepBackboneTrainingResult, error)
}

func (b *ResidentTrainingBackend) stepBackboneEnabled() bool {
	return b != nil && (b.stepResultsEnabled() || b.Scope == "ptx-resident-stepbackbone-hybrid-v1" || b.Scope == "vulkan-resident-stepbackbone-hybrid-v1")
}
func (m *FlowLMTrainingCPU) BuildStepBackboneTrainingTape(weight *LSDWeightMLP, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs, maxBytes int64) (StepBackboneTrainingTape, error) {
	return m.buildStepBackboneTrainingTape(weight, batch, hidden, in, cfg, samples, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildStepBackboneTrainingTape(weight *LSDWeightMLP, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs, maxBytes int64, borrow bool) (StepBackboneTrainingTape, error) {
	var result StepBackboneTrainingTape
	if m == nil || in.Frames != batch.Frames || in.Hidden != m.Hidden || in.Latent != m.LatentDim {
		return result, fmt.Errorf("step backbone reduction topology")
	}
	step, err := weight.buildStepWeightTrainingTape(in, cfg, samples, maxBytes, borrow)
	if err != nil {
		return result, err
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	prefix := rows - batch.Frames
	// Model dimensions/batch are admitted by the backbone builder before use;
	// keep host seed placeholder bounded by the conditioner sequence ceiling.
	if rows < 1 || rows > 1024 || m.Hidden < 1 || m.Hidden > 16384 {
		return result, fmt.Errorf("step backbone rows")
	}
	backbone, err := m.buildEOSBackboneTrainingTape(batch, make([]float32, rows*m.Hidden), hidden, in.Mask, cfg.EOSLossWeight, maxBytes, borrow)
	if err != nil {
		return result, err
	}
	t := step.Tape
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	// Prefix derivatives are zero; preserve the row-wise host copy via a pad.
	g := &flowTapeGraph{tape: t, params: map[string]int{}, adj: map[int]int{}}
	padded := g.command("vector", []int{step.Tape.Outputs[0], g.zero(1)}, []int{rows * m.Hidden}, 0, 2, prefix*m.Hidden, 0, 0)
	seed := g.reshape(padded, rows, m.Hidden)
	t = g.tape
	offset := len(t.Tensors)
	t.Tensors = append(t.Tensors, backbone.Tape.Tensors...)
	mapping := make([]int, len(backbone.Tape.Tensors))
	for i := range mapping {
		mapping[i] = offset + i
	}
	mapping[backbone.seedID] = seed
	for _, c := range backbone.Tape.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		for i, id := range c.Inputs {
			c.Inputs[i] = mapping[id]
		}
		c.Output = mapping[c.Output]
		t.Commands = append(t.Commands, c)
	}
	index := 2
	voice := -1
	if batch.VoiceFrames > 0 {
		voice = mapping[backbone.Tape.Outputs[index]]
		index++
	}
	normalized := mapping[backbone.Tape.Outputs[index]]
	g = &flowTapeGraph{tape: t, params: map[string]int{}, adj: map[int]int{}}
	final := g.command("vector", []int{normalized, step.Tape.Outputs[1]}, []int{batch.Frames * m.LatentDim}, 0, 0, 0, 1, 0)
	t = g.tape
	t.Outputs = append([]int(nil), step.Tape.Outputs[:5]...)
	if voice >= 0 {
		t.Outputs = append(t.Outputs, voice)
	}
	t.Outputs = append(t.Outputs, final)
	bindings := map[string]train.ResidentTapeBinding{}
	for name, b := range step.ResidentBindings {
		bindings[name] = b
	}
	for name, b := range backbone.ResidentBindings {
		if _, exists := bindings[name]; exists {
			return result, fmt.Errorf("step backbone duplicate parameter")
		}
		bindings[name] = train.ResidentTapeBinding{Parameter: mapping[b.Parameter], Gradient: mapping[b.Gradient]}
	}
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		t.Outputs = append(t.Outputs, bindings[name].Gradient)
	}
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return result, err
	}
	return StepBackboneTrainingTape{t, bindings}, nil
}
func decodeStepBackboneBoundary(out [][]float32, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, samples WeightSampleInputs) (stepBackboneTrainingResult, error) {
	var r stepBackboneTrainingResult
	count := 6
	if batch.VoiceFrames > 0 {
		count = 7
	}
	if len(out) != count {
		return r, fmt.Errorf("step backbone boundary count")
	}
	reduction, err := decodeStepWeightBoundary(out[:5], in, samples)
	if err != nil {
		return r, err
	}
	r.Reduction = reduction
	index := 5
	r.Inputs.VoiceLatents = make([]float32, len(batch.VoiceLatents))
	if batch.VoiceFrames > 0 {
		r.Inputs.VoiceLatents = out[index]
		index++
	}
	r.Inputs.NormalizedLatents = out[index]
	values := [][]float32{r.Inputs.VoiceLatents, r.Inputs.NormalizedLatents}
	sizes := []int{len(batch.VoiceLatents), len(batch.NormalizedLatents)}
	sources := [][]float32{batch.VoiceLatents, batch.NormalizedLatents, hidden, in.DiagonalCondition, in.DistillCondition, in.DiagonalTarget, in.DistillTarget, in.DiagonalLogvar, in.DistillLogvar, in.RowMetrics, samples.DiagonalTime, samples.DistillS, samples.DistillT, samples.DiagonalSeed, samples.DistillSeed}
	for i, x := range values {
		if len(x) != sizes[i] || !finiteF32(x) {
			return stepBackboneTrainingResult{}, fmt.Errorf("step backbone input result shape/nonfinite")
		}
		for _, y := range append(values[:i], out[:5]...) {
			if f32SlicesOverlap(x, y) {
				return stepBackboneTrainingResult{}, fmt.Errorf("step backbone overlapping outputs")
			}
		}
		for _, source := range sources {
			if f32SlicesOverlap(x, source) {
				return stepBackboneTrainingResult{}, fmt.Errorf("step backbone borrowed input")
			}
		}
	}
	// Conditioner final normalized-row VJP is zero; preserve only the direct
	// target term at that row after the final addition, including padding.
	start := (batch.Frames - 1) * in.Latent
	for i, x := range r.Inputs.NormalizedLatents[start:] {
		if x != r.Reduction.DTarget[start+i] {
			return stepBackboneTrainingResult{}, fmt.Errorf("step backbone final normalized row")
		}
	}
	return r, nil
}

// Recompute independently pruned root groups to bound later-phase gradient
// storage. The full source and complete inventory are admitted for every group;
// sorted disjoint groups accumulate each named gradient exactly once.
func stepBackboneGradientGroupSize(source train.Tape) int {
	for _, c := range source.Commands {
		if c.Op == "attention-forward" && len(c.Inputs) > 0 {
			shape := source.Tensors[c.Inputs[0]].Shape
			if len(shape) > 0 && shape[0] > 128 {
				return 1
			}
		}
	}
	return 8
}

func (b *ResidentTrainingBackend) runGroupedStepBackbone(source train.Tape, inventory map[string]train.ResidentTapeBinding, outputs []int) ([][]float32, error) {
	if b.state == nil {
		return nil, fmt.Errorf("grouped backbone requires resident owner")
	}
	tape, bindings, compact, err := compactResidentEmbeddingTape(source, inventory, b.MaxTapeBytes)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	groupSize := stepBackboneGradientGroupSize(source)
	var values [][]float32
	for start := 0; start < len(names); start += groupSize {
		var roots []int
		if start == 0 {
			roots = outputs
		}
		var got [][]float32
		if b.handoffLease != nil {
			got, err = b.handoffLease.RunGradientGroupTape(b.ctx, tape, bindings, compact, roots, names[start:min(start+groupSize, len(names))], 1)
		} else {
			got, err = b.state.RunGradientGroupTape(b.ctx, tape, bindings, compact, roots, names[start:min(start+groupSize, len(names))], 1, b.MaxTapeBytes)
		}
		if err != nil {
			return nil, err
		}
		if start == 0 {
			values = got
		}
	}
	return values, nil
}

func (b *ResidentTrainingBackend) evaluateStepBackbone(lm *FlowLMTrainingCPU, flow *FlowHeadCPU, weight *LSDWeightMLP, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs) (stepBackboneTrainingResult, error) {
	var r stepBackboneTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.stepBackboneEnabled() || b.trainer.FlowLM != lm || b.trainer.Flow != flow || b.trainer.Weighting != weight || flow.StepReductionBackend != b || weight.TrainingBackend != b {
		return r, b.fail(fmt.Errorf("step backbone owner/scope"))
	}
	if b.stagedBackboneEnabled(batch) {
		return b.evaluateStagedStepBackbone(lm, flow, weight, batch, hidden, in, cfg, samples)
	}
	p, err := lm.buildStepBackboneTrainingTape(weight, batch, hidden, in, cfg, samples, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	count := 6
	if batch.VoiceFrames > 0 {
		count = 7
	}
	outputs := p.Tape.Outputs[:count]
	if b.stepResultsEnabled() {
		p, err = buildStepResultsTape(p, batch, in, b.MaxTapeBytes)
		if err != nil {
			return r, b.fail(err)
		}
		outputs = stepResultsOutputs(p, batch)
	}
	var out [][]float32
	if b.sharedWorkspaceEnabled() {
		out, err = b.runGroupedStepBackbone(p.Tape, p.ResidentBindings, outputs)
	} else {
		out, err = b.runResidentTape(p.Tape, p.ResidentBindings, outputs, true, 1)
	}
	if err != nil {
		return r, b.fail(err)
	}
	if b.stepResultsEnabled() {
		r, err = decodeStepResultsBoundary(out, batch, hidden, in, samples)
	} else {
		r, err = decodeStepBackboneBoundary(out, batch, hidden, in, samples)
	}
	if err != nil {
		return stepBackboneTrainingResult{}, b.fail(err)
	}
	return r, nil
}
