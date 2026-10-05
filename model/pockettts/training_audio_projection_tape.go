package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type AudioProjectionTrainingResult struct{ Output, DNormalized []float32 }
type TrainingAudioProjectionBackend interface {
	AudioProjectionEnabled() bool
	EvaluateAudioProjection(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, bool) (AudioProjectionTrainingResult, error)
}

func (b *ResidentTrainingBackend) AudioProjectionEnabled() bool {
	return b != nil && (b.VoiceSequenceEnabled() || b.Scope == "ptx-resident-audiofused-hybrid-v1" || b.Scope == "vulkan-resident-audiofused-hybrid-v1")
}

// BuildAudioProjectionTrainingTape retains BOS/shifted audio and affine VJPs
// within one graph. The final normalized-latent row is unused and has zero VJP.
func (m *FlowLMTrainingCPU) BuildAudioProjectionTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64) (SequenceTrainingTape, error) {
	return m.buildAudioProjectionTrainingTape(batch, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildAudioProjectionTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64, borrowParameters bool) (SequenceTrainingTape, error) {
	if m == nil || m.LatentDim < 1 || m.LatentDim > 16384 || m.Hidden < 1 || m.Hidden > 16384 || batch.Frames < 1 || batch.Frames > 1024 || m.Input.In != m.LatentDim || m.Input.Out != m.Hidden || m.Input.Bias != nil || len(m.BOS) != m.LatentDim || len(batch.NormalizedLatents) != batch.Frames*m.LatentDim || len(seed) != batch.Frames*m.Hidden || !finiteF32(m.BOS) || !finiteF32(batch.NormalizedLatents) || !finiteF32(seed) {
		return SequenceTrainingTape{}, fmt.Errorf("fused audio projection input/shape")
	}
	if err := validateOwnedTrainingLinear(m.Input, false); err != nil {
		return SequenceTrainingTape{}, err
	}
	rows, c, h := batch.Frames, m.LatentDim, m.Hidden
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	bos := g.param("flow_lm.bos_emb", []int{c}, m.BOS)
	normalized := g.constant([]int{rows * c}, batch.NormalizedLatents)
	shifted := g.pad(bos, 0, rows*c)
	if rows > 1 {
		shifted = g.sum(shifted, g.pad(g.slice(normalized, 0, (rows-1)*c), c, rows*c), rows*c)
	}
	x := g.reshape(shifted, rows, c)
	w := g.param("flow_lm.input_linear.weight", []int{h, c}, m.Input.Weight)
	y := g.command("linear-forward", []int{x, g.zero(rows, c), w, g.zero(h)}, []int{rows, h}, 0)
	dy := g.constant([]int{rows, h}, seed)
	operands := []int{x, g.zero(rows, c), w, dy, g.zero(rows, h)}
	dx := g.command("linear-backward", operands, []int{rows, c}, 0)
	dw := g.command("linear-backward", operands, []int{h, c}, 2)
	dbos := g.slice(dx, 0, c)
	dn := g.zero(rows * c)
	if rows > 1 {
		dn = g.pad(g.slice(dx, c, (rows-1)*c), 0, rows*c)
	}
	g.tape.Outputs = []int{g.flat(y), dn, g.flat(dw), dbos}
	bindings := map[string]train.ResidentTapeBinding{"flow_lm.input_linear.weight": {Parameter: w, Gradient: g.tape.Outputs[2]}, "flow_lm.bos_emb": {Parameter: bos, Gradient: dbos}}
	return materializeSequenceTape(g, bindings, maxBytes)
}
func validateAudioProjectionBoundary(r AudioProjectionTrainingResult, frames, hidden, latent int, inputs ...[]float32) error {
	if len(r.Output) != frames*hidden || len(r.DNormalized) != frames*latent || !finiteF32(r.Output) || !finiteF32(r.DNormalized) || f32SlicesOverlap(r.Output, r.DNormalized) {
		return fmt.Errorf("fused audio projection result shape/nonfinite")
	}
	for _, x := range inputs {
		if f32SlicesOverlap(r.Output, x) || f32SlicesOverlap(r.DNormalized, x) {
			return fmt.Errorf("fused audio borrowed input")
		}
	}
	for _, v := range r.DNormalized[(frames-1)*latent:] {
		if v != 0 {
			return fmt.Errorf("fused audio terminal row VJP")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateAudioProjection(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed []float32, accumulate bool) (AudioProjectionTrainingResult, error) {
	var r AudioProjectionTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.AudioProjectionEnabled() || m != b.trainer.FlowLM || m.SequenceBackend != b || m.ProjectionBackend != b {
		return r, b.fail(fmt.Errorf("fused audio owner/scope mismatch"))
	}
	p, err := m.buildAudioProjectionTrainingTape(batch, seed, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	outputs := p.Tape.Outputs[:2]
	if accumulate {
		outputs = p.Tape.Outputs[1:2]
	} else if zeroTrainingSeed(seed) {
		outputs = p.Tape.Outputs[:1]
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return r, b.fail(err)
	}
	if accumulate {
		r.Output = make([]float32, batch.Frames*m.Hidden)
		r.DNormalized = out[0]
	} else {
		r.Output = out[0]
		r.DNormalized = make([]float32, batch.Frames*m.LatentDim)
		if len(out) > 1 {
			r.DNormalized = out[1]
		}
	}
	if err = validateAudioProjectionBoundary(r, batch.Frames, m.Hidden, m.LatentDim, batch.NormalizedLatents, seed); err != nil {
		return AudioProjectionTrainingResult{}, b.fail(err)
	}
	return r, nil
}
