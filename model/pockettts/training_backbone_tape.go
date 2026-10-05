package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

// TrainingBackboneBackend composes conditioner and transformer edges within one
// tape. It preserves the public sequence, but does not export its VJP boundary.
type TrainingBackboneBackend interface {
	BackboneEnabled() bool
	EvaluateBackbone(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, bool) (BackboneTrainingResult, error)
}
type BackboneTrainingResult struct{ Sequence, Output, DVoice, DNormalized []float32 }
type BackboneTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
	seedID           int // explicit composition edge, standalone seed arithmetic unchanged
}

func (b *ResidentTrainingBackend) BackboneEnabled() bool {
	return b != nil && (b.EOSBackboneEnabled() || b.Scope == "ptx-resident-backbone-hybrid-v1" || b.Scope == "vulkan-resident-backbone-hybrid-v1")
}

// BuildBackboneTrainingTape owns its public source constants. The private
// synchronous resident builder borrows named parameters only. Full source graph
// admission precedes pruning; no cross-tape cache or storage reuse is introduced.
// Outputs are sequence, transformed rows, optional raw voice VJP, normalized
// latent VJP, then all named gradients in sorted order.
func (m *FlowLMTrainingCPU) BuildBackboneTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64) (BackboneTrainingTape, error) {
	return m.buildBackboneTrainingTape(batch, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildBackboneTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64, borrow bool) (BackboneTrainingTape, error) {
	return m.buildBackboneTrainingTapeSeed(batch, seed, maxBytes, borrow, nil)
}
func (m *FlowLMTrainingCPU) buildBackboneTrainingTapeSeed(batch FlowLMTrainingBatch, seed []float32, maxBytes int64, borrow bool, seedGraph func(*transformerTapeGraph, int, int) (int, map[string]train.ResidentTapeBinding)) (BackboneTrainingTape, error) {
	var result BackboneTrainingTape
	if maxBytes <= 0 {
		return result, fmt.Errorf("backbone tape byte ceiling")
	}
	if err := m.validateConditionerTrainingTape(batch, seed); err != nil {
		return result, err
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	// The graph-derived sequence is not available on host. Validate the topology
	// with the admitted same-shape finite seed; admit every source constant below.
	if err := validateTrainableTransformer(m.Transformer, seed, seed, rows); err != nil {
		return result, err
	}
	if m.Transformer.Width != m.Hidden {
		return result, fmt.Errorf("backbone transformer width")
	}
	if err := preflightTransformerTrainingTape(m.Transformer, rows); err != nil {
		return result, err
	}
	g := &transformerTapeGraph{borrowParameters: borrow, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	cache := m.appendConditionerForwardGraph(g, batch)
	dy := g.constant([]int{rows, m.Hidden}, seed)
	extra := map[string]train.ResidentTapeBinding{}
	var hook func(int, int) int
	if seedGraph != nil {
		hook = func(output, seed int) int { var next int; next, extra = seedGraph(g, output, seed); return next }
	}
	output, dsequence, gradients, err := m.Transformer.appendTransformerTrainingGraphSeed(g, g.reshape(cache.sequence, rows, m.Hidden), dy, rows, hook)
	if err != nil {
		return result, err
	}
	inputs, bindings := m.appendConditionerBackwardGraph(g, batch, cache, dsequence)
	for name, binding := range extra {
		bindings[name] = binding
	}
	for name, id := range g.params {
		if _, ok := bindings[name]; ok {
			continue
		}
		gradient, ok := gradients[id]
		if !ok {
			return result, fmt.Errorf("missing backbone gradient %s", name)
		}
		bindings[name] = train.ResidentTapeBinding{Parameter: id, Gradient: gradient}
	}
	g.tape.Outputs = append([]int{cache.sequence, output}, inputs...)
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		g.tape.Outputs = append(g.tape.Outputs, bindings[name].Gradient)
	}
	if len(g.tape.Tensors) > 4096 || len(g.tape.Commands) > 4096 {
		return result, fmt.Errorf("backbone graph ceiling")
	}
	materialized, err := materializeSequenceTape(g, bindings, maxBytes)
	if err != nil {
		return result, err
	}
	if err = train.ValidateResidentTapeBindings(materialized.Tape, bindings, maxBytes); err != nil {
		return result, err
	}
	return BackboneTrainingTape{Tape: materialized.Tape, ResidentBindings: bindings, seedID: dy}, nil
}
func validateBackboneResult(r BackboneTrainingResult, m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed []float32) error {
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	values := [][]float32{r.Sequence, r.Output, r.DVoice, r.DNormalized}
	sizes := []int{rows * m.Hidden, rows * m.Hidden, batch.VoiceFrames * m.LatentDim, batch.Frames * m.LatentDim}
	for i, x := range values {
		if len(x) != sizes[i] || !finiteF32(x) {
			return fmt.Errorf("backbone output%d shape/nonfinite", i)
		}
		for _, source := range [][]float32{batch.VoiceLatents, batch.NormalizedLatents, seed} {
			if f32SlicesOverlap(x, source) {
				return fmt.Errorf("backbone borrowed output%d", i)
			}
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("backbone aliased outputs")
			}
		}
	}
	for _, v := range r.DNormalized[(batch.Frames-1)*m.LatentDim:] {
		if v != 0 {
			return fmt.Errorf("backbone final normalized VJP")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateBackbone(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed []float32, accumulate bool) (BackboneTrainingResult, error) {
	var r BackboneTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.BackboneEnabled() || b.trainer.FlowLM != m || m.SequenceBackend != b || m.ProjectionBackend != b || m.TransformerBackend != b {
		return r, b.fail(fmt.Errorf("backbone owner/scope"))
	}
	p, err := m.buildBackboneTrainingTape(batch, seed, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	count := 3
	if batch.VoiceFrames > 0 {
		count = 4
	}
	outputs := p.Tape.Outputs[:count]
	forward := !accumulate && zeroTrainingSeed(seed)
	if accumulate {
		outputs = outputs[2:]
	} else if forward {
		outputs = outputs[:2]
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r = BackboneTrainingResult{Sequence: make([]float32, len(seed)), Output: make([]float32, len(seed)), DVoice: make([]float32, len(batch.VoiceLatents)), DNormalized: make([]float32, len(batch.NormalizedLatents))}
	index := 0
	if !accumulate {
		r.Sequence, r.Output = out[0], out[1]
		index = 2
	}
	if !forward {
		if batch.VoiceFrames > 0 {
			r.DVoice = out[index]
			index++
		}
		r.DNormalized = out[index]
	}
	if err = validateBackboneResult(r, m, batch, seed); err != nil {
		return BackboneTrainingResult{}, b.fail(err)
	}
	return r, nil
}
