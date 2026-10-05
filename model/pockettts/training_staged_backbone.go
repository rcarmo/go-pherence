package pockettts

import (
	"fmt"
	"slices"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private stage route keeps the existing shared-workspace identity and limits.
// Larger sequences use separately declared, fully admitted sources; the CLI
// continues its conservative admission and never falls back after a rejection.
func (b *ResidentTrainingBackend) stagedBackboneEnabled(batch FlowLMTrainingBatch) bool {
	return b != nil && b.sharedWorkspaceEnabled() && 1+batch.VoiceFrames+len(batch.TextTokens)+batch.Frames > 128
}

type stagedBackboneBoundary struct {
	model  *FlowLMTrainingCPU
	batch  FlowLMTrainingBatch
	mask   []bool
	weight float32
	hidden []float32
}

func (b *ResidentTrainingBackend) evaluateStagedForward(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, weight float32) (*flowLMTrainingTape, float64, error) {
	if err := b.admitEOSBackbone(m); err != nil {
		return nil, 0, err
	}
	if b.stagedBackbone != nil || b.transformerStages != nil || b.handoffLease != nil {
		return nil, 0, b.fail(fmt.Errorf("staged backbone already active"))
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	zero := make([]float32, rows*m.Hidden)
	// Complete model/stage source admission occurs before first execution.
	if err := validateTransformerStageModel(m.Transformer, zero, zero, rows); err != nil {
		return nil, 0, b.fail(err)
	}
	conditioner, err := m.buildConditionerTrainingTape(batch, zero, b.MaxTapeBytes, true)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	conditionerRoots := append([]int(nil), conditioner.Tape.Outputs...)
	if err = b.admitStagedStorage(conditioner.Tape, conditioner.ResidentBindings, conditionerRoots, false); err != nil {
		return nil, 0, b.fail(err)
	}
	var pool *trainingZeroPool
	if ws := b.forwardZeroWorkspace; ws != nil {
		if ws.zeroGradientOwner != b || !ws.compatible(m, b.trainer.Flow) {
			return nil, 0, b.fail(fmt.Errorf("staged workspace provenance"))
		}
		pool = ws.zeroGradientPool
	}
	for i := range m.Transformer.Layers {
		forward, e := buildTransformerLayerStage(m.Transformer, i, zero, zero, rows, true, b.MaxTapeBytes, pool)
		if e != nil {
			return nil, 0, b.fail(e)
		}
		if e = b.admitStagedStorage(forward.Tape, forward.ResidentBindings, forward.Tape.Outputs[:1], false); e != nil {
			return nil, 0, b.fail(e)
		}
		reverse, e := buildTransformerLayerStage(m.Transformer, i, zero, zero, rows, false, b.MaxTapeBytes)
		if e != nil {
			return nil, 0, b.fail(e)
		}
		if e = b.admitStagedStorage(reverse.Tape, reverse.ResidentBindings, reverse.Tape.Outputs[1:2], true); e != nil {
			return nil, 0, b.fail(e)
		}
	}
	if m.Transformer.FinalWeight != nil {
		reverse, e := buildTransformerFinalNormStage(m.Transformer, zero, zero, rows, false, b.MaxTapeBytes)
		if e != nil {
			return nil, 0, b.fail(e)
		}
		if e = b.admitStagedStorage(reverse.Tape, reverse.ResidentBindings, reverse.Tape.Outputs[1:2], true); e != nil {
			return nil, 0, b.fail(e)
		}
	}
	eosPreflight, e := m.buildEOSProjectionTrainingTape(make([]float32, batch.Frames*m.Hidden), mask, weight, b.MaxTapeBytes, true)
	if e != nil {
		return nil, 0, b.fail(e)
	}
	if e = b.admitStagedStorage(eosPreflight.Tape, eosPreflight.ResidentBindings, eosPreflight.Tape.Outputs, false); e != nil {
		return nil, 0, b.fail(e)
	}
	values, err := b.runResidentTape(conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs[:1], false, 1)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	if len(values) != 1 || len(values[0]) != len(zero) || !finiteF32(values[0]) {
		return nil, 0, b.fail(fmt.Errorf("staged conditioner boundary"))
	}
	transformed, err := b.beginTransformerStages(m.Transformer, values[0], rows)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	hidden := append([]float32(nil), transformed[(rows-batch.Frames)*m.Hidden:]...)
	// Current absolute-dot overflow admission applies to the actual hidden rows.
	eos, err := m.buildEOSProjectionTrainingTape(hidden, mask, weight, b.MaxTapeBytes, true)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	if err = b.admitStagedStorage(eos.Tape, eos.ResidentBindings, eos.Tape.Outputs[:1], false); err != nil {
		return nil, 0, b.fail(err)
	}
	means, err := b.runResidentTape(eos.Tape, eos.ResidentBindings, eos.Tape.Outputs[:1], false, 1)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	if len(means) != 1 || len(means[0]) != 1 || !isFinite(means[0][0]) || means[0][0] < 0 {
		return nil, 0, b.fail(fmt.Errorf("staged EOS mean"))
	}
	loss := float64(means[0][0])
	ticket := &stagedBackboneBoundary{model: m, batch: batch, mask: append([]bool(nil), mask...), weight: weight, hidden: append([]float32(nil), hidden...)}
	ticket.batch.TextTokens = append([]uint32(nil), batch.TextTokens...)
	ticket.batch.VoiceLatents = append([]float32(nil), batch.VoiceLatents...)
	ticket.batch.NormalizedLatents = append([]float32(nil), batch.NormalizedLatents...)
	b.stagedBackbone = ticket
	// Pin only the admitted computed hidden boundary for row handoffs. Model
	// sources have already been validated by every complete declared stage.
	g := &flowTapeGraph{}
	source := g.constant(hidden)
	out := g.command("vector", []int{source, g.zero(1)}, []int{len(hidden)}, 0, 1, 0, 0, 0)
	g.tape.Outputs = []int{out}
	b.handoffLease, err = b.state.BeginZeroPrunedTapeLease(b.ctx, g.tape, nil, []int{out}, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	b.handoffHidden = hidden
	b.handoffRoot = out
	b.handoffRow = -1
	b.handoffFrames = batch.Frames
	b.handoffWidth = m.Hidden
	output := FlowLMTrainingOutput{Z: hidden, PrefixRows: rows - batch.Frames, SequenceRows: rows}
	tape := &flowLMTrainingTape{rows: rows, prefix: rows - batch.Frames, textOffset: 1 + batch.VoiceFrames, transformerBackend: m.TransformerBackend, projectionBackend: m.ProjectionBackend, sequenceBackend: m.SequenceBackend, output: output, fusedEOS: true, backbone: b, eosBackbone: b, eosMask: append([]bool(nil), mask...), eosWeight: weight}
	return tape, loss, nil
}

func (b *ResidentTrainingBackend) evaluateStagedStepBackbone(lm *FlowLMTrainingCPU, flow *FlowHeadCPU, weight *LSDWeightMLP, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, cfg TrainingStepConfig, samples WeightSampleInputs) (stepBackboneTrainingResult, error) {
	var result stepBackboneTrainingResult
	if err := b.admitGraph(); err != nil {
		return result, err
	}
	ticket := b.stagedBackbone
	transformer := b.transformerStages
	if ticket == nil || transformer == nil || ticket.model != lm || lm != b.trainer.FlowLM || flow != b.trainer.Flow || weight != b.trainer.Weighting || ticket.weight != cfg.EOSLossWeight || !slices.Equal(ticket.mask, in.Mask) || !slices.Equal(ticket.hidden, hidden) || !slices.Equal(ticket.batch.TextTokens, batch.TextTokens) || !slices.Equal(ticket.batch.VoiceLatents, batch.VoiceLatents) || !slices.Equal(ticket.batch.NormalizedLatents, batch.NormalizedLatents) || ticket.batch.Frames != batch.Frames || ticket.batch.VoiceFrames != batch.VoiceFrames || b.handoffLease != nil {
		return result, b.fail(fmt.Errorf("staged backbone backward provenance"))
	}
	if !sameResidentLiveBoundary(b.host, b.trainer) {
		return result, b.fail(fmt.Errorf("staged parameter boundary changed"))
	}
	// Validate all reduction/EOS/conditioner sources before the first accumulation.
	step, err := weight.buildStepWeightTrainingTape(in, cfg, samples, b.MaxTapeBytes, true)
	if err != nil {
		return result, b.fail(err)
	}
	eos, err := lm.buildEOSProjectionTrainingTape(hidden, in.Mask, cfg.EOSLossWeight, b.MaxTapeBytes, true)
	if err != nil {
		return result, b.fail(err)
	}
	rows := transformer.rows
	prefix := rows - batch.Frames
	conditionerPreflight, e := lm.buildConditionerTrainingTape(batch, make([]float32, rows*lm.Hidden), b.MaxTapeBytes, true)
	if e != nil {
		return result, b.fail(e)
	}
	for _, stage := range []struct {
		tape     train.Tape
		bindings map[string]train.ResidentTapeBinding
	}{{step.Tape, step.ResidentBindings}, {eos.Tape, eos.ResidentBindings}, {conditionerPreflight.Tape, conditionerPreflight.ResidentBindings}} {
		if e = b.admitStagedStorage(stage.tape, stage.bindings, stage.tape.Outputs, false); e != nil {
			return result, b.fail(e)
		}
	}
	values, err := b.runResidentTape(step.Tape, step.ResidentBindings, step.Tape.Outputs[:5], true, 1)
	if err != nil {
		return result, b.fail(err)
	}
	reduction, err := decodeStepWeightBoundary(values, in, samples)
	if err != nil {
		return result, b.fail(err)
	}
	eosValues, err := b.runResidentTape(eos.Tape, eos.ResidentBindings, eos.Tape.Outputs[:2], true, 1)
	if err != nil {
		return result, b.fail(err)
	}
	if len(eosValues) != 2 || len(eosValues[0]) != 1 || len(eosValues[1]) != len(hidden) || !finiteF32(eosValues[1]) {
		return result, b.fail(fmt.Errorf("staged EOS VJP boundary"))
	}
	// The original backbone computes padded flow seed + padded EOS input VJP.
	// Preserve that F32 association using the existing validated vector operation.
	eosSeed := make([]float32, rows*lm.Hidden)
	copy(eosSeed[prefix*lm.Hidden:], eosValues[1])
	flowSeed := make([]float32, rows*lm.Hidden)
	copy(flowSeed[prefix*lm.Hidden:], reduction.DCondition)
	n := len(eosSeed)
	seedTape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{n}, Values: flowSeed}, {Shape: []int{n}, Values: eosSeed}, {Shape: []int{n}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	summed, err := b.runUtilityTape(seedTape, seedTape.Outputs)
	if err != nil {
		return result, b.fail(err)
	}
	dx, err := b.backwardTransformerStages(lm.Transformer, transformer.boundaries[0], summed[0], rows)
	if err != nil {
		return result, b.fail(err)
	}
	conditioner, err := lm.buildConditionerTrainingTape(batch, dx, b.MaxTapeBytes, true)
	if err != nil {
		return result, b.fail(err)
	}
	count := 2
	if batch.VoiceFrames > 0 {
		count = 3
	}
	// The actual VJP can change immutable-zero slot sharing; admit its complete
	// source before execution rather than relying on the zero-seed forecast.
	if err = b.admitStagedStorage(conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs, false); err != nil {
		return result, b.fail(err)
	}
	inputs, err := b.runResidentTape(conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs[1:count], true, 1)
	if err != nil {
		return result, b.fail(err)
	}
	normalized := inputs[len(inputs)-1]
	final, err := b.EvaluateStepInputGradient(flow, normalized, reduction.DTarget)
	if err != nil {
		return result, b.fail(err)
	}
	out := append([][]float32(nil), values...)
	if batch.VoiceFrames > 0 {
		out = append(out, inputs[0])
	}
	out = append(out, final)
	result, err = decodeStepBackboneBoundary(out, batch, hidden, in, samples)
	if err != nil {
		return result, b.fail(err)
	}
	if b.stepResultsEnabled() {
		result.Reduction.DCondition = nil
		result.Reduction.DTarget = nil
	}
	b.stagedBackbone = nil
	return result, nil
}
