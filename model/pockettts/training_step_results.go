package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Private step result policy. Public reduction and backbone builders keep their
// full output contracts; only the step discards unused condition/direct arrays.
func (b *ResidentTrainingBackend) stepResultsEnabled() bool {
	return b != nil && (b.distillResultsEnabled() || b.Scope == "ptx-resident-stepresults-hybrid-v1" || b.Scope == "vulkan-resident-stepresults-hybrid-v1")
}

func buildStepResultsTape(p StepBackboneTrainingTape, batch FlowLMTrainingBatch, in StepReductionInputs, maxBytes int64) (StepBackboneTrainingTape, error) {
	// Preserve complete source admission before adding a bounded direct-tail root.
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return StepBackboneTrainingTape{}, err
	}
	count := 6
	if batch.VoiceFrames > 0 {
		count = 7
	}
	if batch.Frames < 1 || in.Latent < 1 || len(p.Tape.Outputs) < count {
		return StepBackboneTrainingTape{}, fmt.Errorf("step results source topology")
	}
	t := p.Tape
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	g := &flowTapeGraph{tape: t, params: map[string]int{}, adj: map[int]int{}}
	tail := g.command("vector", []int{t.Outputs[1], g.zero(1)}, []int{in.Latent}, 0, 1, (batch.Frames-1)*in.Latent, 0, 0)
	t = g.tape
	t.Outputs = append([]int(nil), p.Tape.Outputs...)
	t.Outputs = append(t.Outputs, tail)
	if err := train.ValidateResidentTapeBindings(t, p.ResidentBindings, maxBytes); err != nil {
		return StepBackboneTrainingTape{}, err
	}
	return StepBackboneTrainingTape{t, p.ResidentBindings}, nil
}
func stepResultsOutputs(p StepBackboneTrainingTape, batch FlowLMTrainingBatch) []int {
	count := 6
	if batch.VoiceFrames > 0 {
		count = 7
	}
	// Scaled diagonal/distill logvars,metrics, optional voice, final normalized
	// gradient, and the direct-tail witness. No fake full derivative placeholders.
	ids := append([]int(nil), p.Tape.Outputs[2:count]...)
	return append(ids, p.Tape.Outputs[len(p.Tape.Outputs)-1])
}
func decodeStepResultsBoundary(out [][]float32, batch FlowLMTrainingBatch, hidden []float32, in StepReductionInputs, samples WeightSampleInputs) (stepBackboneTrainingResult, error) {
	var r stepBackboneTrainingResult
	count := 5
	if batch.VoiceFrames > 0 {
		count = 6
	}
	if len(out) != count || len(out[2]) != 5 {
		return r, fmt.Errorf("step results boundary count")
	}
	sizes := []int{in.Frames, in.Frames, 5}
	if batch.VoiceFrames > 0 {
		sizes = append(sizes, len(batch.VoiceLatents))
	}
	sizes = append(sizes, len(batch.NormalizedLatents), in.Latent)
	sources := [][]float32{batch.VoiceLatents, batch.NormalizedLatents, hidden, in.DiagonalCondition, in.DistillCondition, in.DiagonalTarget, in.DistillTarget, in.DiagonalLogvar, in.DistillLogvar, in.RowMetrics, samples.DiagonalTime, samples.DistillS, samples.DistillT, samples.DiagonalSeed, samples.DistillSeed}
	for i, x := range out {
		if len(x) != sizes[i] || !finiteF32(x) {
			return r, fmt.Errorf("step results shape/nonfinite")
		}
		for _, y := range out[:i] {
			if f32SlicesOverlap(x, y) {
				return r, fmt.Errorf("step results aliased output")
			}
		}
		for _, source := range sources {
			if f32SlicesOverlap(x, source) {
				return r, fmt.Errorf("step results borrowed output")
			}
		}
	}
	for row, active := range in.Mask {
		if !active && (out[0][row] != 0 || out[1][row] != 0) {
			return r, fmt.Errorf("step results inactive logvar")
		}
	}
	metrics := TrainingStepMetrics{FlowDiagonal: float64(out[2][0]), FlowDistill: float64(out[2][1]), FlowLoss: float64(out[2][2]), EOS: float64(out[2][3]), Loss: float64(out[2][4])}
	if metrics.FlowDiagonal < 0 || metrics.FlowDistill < 0 || metrics.EOS < 0 {
		return r, fmt.Errorf("step results negative square/EOS")
	}
	// Deliberately no DCondition/DTarget arrays: the private step never consumes
	// them. The public reduction API still validates and returns its full arrays.
	r.Reduction = StepReductionResult{DiagonalLogvar: out[0], DistillLogvar: out[1], Metrics: metrics}
	index := 3
	r.Inputs.VoiceLatents = make([]float32, len(batch.VoiceLatents))
	if batch.VoiceFrames > 0 {
		r.Inputs.VoiceLatents = out[index]
		index++
	}
	r.Inputs.NormalizedLatents = out[index]
	tail := out[index+1]
	start := (batch.Frames - 1) * in.Latent
	for i, x := range r.Inputs.NormalizedLatents[start:] {
		if x != tail[i] {
			return stepBackboneTrainingResult{}, fmt.Errorf("step results final normalized row")
		}
	}
	if !in.Mask[len(in.Mask)-1] {
		for _, v := range tail {
			if v != 0 {
				return stepBackboneTrainingResult{}, fmt.Errorf("step results inactive direct tail")
			}
		}
	}
	return r, nil
}
