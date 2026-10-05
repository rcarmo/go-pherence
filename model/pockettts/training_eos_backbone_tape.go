package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingEOSBackboneBackend defers EOS parameter/input VJP until backbone
// backward. The forward mean still has its own scalar-only admission boundary.
type TrainingEOSBackboneBackend interface {
	EOSBackboneEnabled() bool
	EvaluateEOSBackboneLoss(*FlowLMTrainingCPU, []float32, []bool, float32) (float64, error)
	EvaluateEOSBackbone(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, []float32, []bool, float32) (BackboneTrainingResult, error)
}

func (b *ResidentTrainingBackend) EOSBackboneEnabled() bool {
	return b != nil && (b.DiagonalEnabled() || b.Scope == "ptx-resident-eosbackbone-hybrid-v1" || b.Scope == "vulkan-resident-eosbackbone-hybrid-v1")
}

// BuildEOSBackboneTrainingTape uses hidden only for the same forward absolute-dot
// overflow admission as standalone EOS. It is not a graph source: the immutable
// backbone reevaluates hidden rows and combines EOS VJP with the caller seed on
// device. Public constants are owned; EOS weight is applied once in mode1.
func (m *FlowLMTrainingCPU) BuildEOSBackboneTrainingTape(batch FlowLMTrainingBatch, seed, hidden []float32, mask []bool, weight float32, maxBytes int64) (BackboneTrainingTape, error) {
	return m.buildEOSBackboneTrainingTape(batch, seed, hidden, mask, weight, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildEOSBackboneTrainingTape(batch FlowLMTrainingBatch, seed, hidden []float32, mask []bool, weight float32, maxBytes int64, borrow bool) (BackboneTrainingTape, error) {
	if len(mask) != batch.Frames {
		return BackboneTrainingTape{}, fmt.Errorf("EOS backbone mask shape")
	}
	labels, active, err := m.admitEOSProjection(hidden, mask, weight, maxBytes)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	return m.buildBackboneTrainingTapeSeed(batch, seed, maxBytes, borrow, func(g *transformerTapeGraph, output, dy int) (int, map[string]train.ResidentTapeBinding) {
		h := m.Hidden
		rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
		prefix := rows - batch.Frames
		x := g.reshape(g.slice(output, prefix*h, batch.Frames*h), batch.Frames, h)
		w := g.param("flow_lm.out_eos.weight", []int{1, h}, m.EOS.Weight)
		bias := g.param("flow_lm.out_eos.bias", []int{1}, m.EOS.Bias)
		logits := g.command("linear-forward", []int{x, g.zero(batch.Frames, h), w, bias}, []int{batch.Frames, 1}, 0)
		label := g.constant([]int{batch.Frames}, labels)
		eosSeed := g.command("eos", []int{g.flat(logits), label}, []int{batch.Frames}, 1)
		c := &g.tape.Commands[len(g.tape.Commands)-1]
		c.Rows, c.Scale = active, weight
		operands := []int{x, g.zero(batch.Frames, h), w, g.reshape(eosSeed, batch.Frames, 1), g.zero(batch.Frames, 1)}
		dx := g.command("linear-backward", operands, []int{batch.Frames, h}, 0)
		dw := g.flat(g.command("linear-backward", operands, []int{1, h}, 2))
		db := g.command("linear-backward", operands, []int{1}, 3)
		next := g.sum(dy, g.pad(dx, prefix*h, rows*h), rows, h)
		return next, map[string]train.ResidentTapeBinding{"flow_lm.out_eos.weight": {Parameter: w, Gradient: dw}, "flow_lm.out_eos.bias": {Parameter: bias, Gradient: db}}
	})
}
func (b *ResidentTrainingBackend) admitEOSBackbone(m *FlowLMTrainingCPU) error {
	if err := b.admitGraph(); err != nil {
		return err
	}
	if !b.EOSBackboneEnabled() || b.trainer.FlowLM != m || m.SequenceBackend != b || m.ProjectionBackend != b || m.TransformerBackend != b || m.EOSBackend != b {
		return b.fail(fmt.Errorf("EOS backbone owner/scope"))
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateEOSBackboneLoss(m *FlowLMTrainingCPU, hidden []float32, mask []bool, weight float32) (float64, error) {
	if err := b.admitEOSBackbone(m); err != nil {
		return 0, err
	}
	p, err := m.buildEOSProjectionTrainingTape(hidden, mask, weight, b.MaxTapeBytes, true)
	if err != nil {
		return 0, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:1], false, 1)
	if err != nil {
		return 0, b.fail(err)
	}
	if len(out) != 1 || len(out[0]) != 1 || !finiteF32(out[0]) || out[0][0] < 0 || f32SlicesOverlap(out[0], hidden) {
		return 0, b.fail(fmt.Errorf("EOS backbone scalar result"))
	}
	return float64(out[0][0]), nil
}
func (b *ResidentTrainingBackend) EvaluateEOSBackbone(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed, hidden []float32, mask []bool, weight float32) (BackboneTrainingResult, error) {
	var r BackboneTrainingResult
	if err := b.admitEOSBackbone(m); err != nil {
		return r, err
	}
	p, err := m.buildEOSBackboneTrainingTape(batch, seed, hidden, mask, weight, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	count := 3
	if batch.VoiceFrames > 0 {
		count = 4
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[2:count], true, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r = BackboneTrainingResult{Sequence: make([]float32, len(seed)), Output: make([]float32, len(seed)), DVoice: make([]float32, len(batch.VoiceLatents)), DNormalized: make([]float32, len(batch.NormalizedLatents))}
	index := 0
	if batch.VoiceFrames > 0 {
		r.DVoice = out[index]
		index++
	}
	r.DNormalized = out[index]
	if err = validateBackboneResult(r, m, batch, seed); err != nil {
		return BackboneTrainingResult{}, b.fail(err)
	}
	if f32SlicesOverlap(r.DVoice, hidden) || f32SlicesOverlap(r.DNormalized, hidden) {
		return BackboneTrainingResult{}, b.fail(fmt.Errorf("EOS backbone borrowed hidden"))
	}
	return r, nil
}
