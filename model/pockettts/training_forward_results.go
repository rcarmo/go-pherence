package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (b *ResidentTrainingBackend) forwardResultsEnabled() bool {
	return b != nil && (b.embeddingCompactEnabled() || b.embeddingIndexEnabled() || b.Scope == "ptx-resident-forwardresults-hybrid-v1" || b.Scope == "vulkan-resident-forwardresults-hybrid-v1")
}
func buildForwardResultsTape(p BackboneTrainingTape, m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, maxBytes int64) (BackboneTrainingTape, error) {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return BackboneTrainingTape{}, err
	}
	if m == nil || batch.Frames < 1 || batch.VoiceFrames < 0 || m.Hidden < 1 || len(p.Tape.Outputs) < 3 {
		return BackboneTrainingTape{}, fmt.Errorf("forward results topology")
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	prefix := rows - batch.Frames
	for _, id := range p.Tape.Outputs[:2] {
		n, _ := p.Tape.Tensors[id].Elements()
		if n != rows*m.Hidden {
			return BackboneTrainingTape{}, fmt.Errorf("forward results source extent")
		}
	}
	g := &transformerTapeGraph{tape: p.Tape, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, borrowParameters: true}
	g.tape.Tensors = append([]train.TensorSpec(nil), p.Tape.Tensors...)
	g.tape.Commands = append([]train.Command(nil), p.Tape.Commands...)
	g.tape.Outputs = append([]int(nil), p.Tape.Outputs...)
	hidden := g.slice(p.Tape.Outputs[1], prefix*m.Hidden, batch.Frames*m.Hidden)
	// Retain checks for every omitted sequence/prefix element; audio hidden is
	// also covered, independently of its own finite boundary validation.
	witness := appendTrainingFiniteWitness(g, p.Tape.Outputs[:2])
	g.tape.Outputs = append(g.tape.Outputs, hidden, witness)
	built, err := materializeSequenceTape(g, p.ResidentBindings, maxBytes)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	p.Tape = built.Tape
	return p, nil
}
func forwardResultsOutputs(p BackboneTrainingTape) []int {
	n := len(p.Tape.Outputs)
	return []int{p.Tape.Outputs[n-2], p.Tape.Outputs[n-3], p.Tape.Outputs[n-1]}
}
func decodeForwardResults(out [][]float32, m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, weight float32, maxBytes int64) (*flowLMTrainingTape, float64, error) {
	if len(out) != 3 {
		return nil, 0, fmt.Errorf("forward results count")
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	prefix := rows - batch.Frames
	sizes := []int{batch.Frames * m.Hidden, 1, 1}
	for i, x := range out {
		if len(x) != sizes[i] || !finiteF32(x) {
			return nil, 0, fmt.Errorf("forward results shape/nonfinite")
		}
		for _, y := range out[:i] {
			if f32SlicesOverlap(x, y) {
				return nil, 0, fmt.Errorf("forward results aliased outputs")
			}
		}
		for _, source := range [][]float32{batch.NormalizedLatents, batch.VoiceLatents} {
			if f32SlicesOverlap(x, source) {
				return nil, 0, fmt.Errorf("forward results borrowed input")
			}
		}
	}
	if out[1][0] < 0 || out[2][0] != 0 {
		return nil, 0, fmt.Errorf("forward results loss/witness")
	}
	if _, _, err := m.admitEOSProjection(out[0], mask, weight, maxBytes); err != nil {
		return nil, 0, err
	}
	backbone, ok := m.SequenceBackend.(TrainingBackboneBackend)
	if !ok {
		return nil, 0, fmt.Errorf("forward results backbone owner")
	}
	eos, ok := m.EOSBackend.(TrainingEOSBackboneBackend)
	if !ok {
		return nil, 0, fmt.Errorf("forward results EOS owner")
	}
	// Private step/backbone policy never consumes the full sequence or EOS logits.
	// Nil fields explicitly describe omitted outputs; no fabricated full arrays.
	output := FlowLMTrainingOutput{Z: out[0], PrefixRows: prefix, SequenceRows: rows}
	return &flowLMTrainingTape{rows: rows, prefix: prefix, textOffset: 1 + batch.VoiceFrames, transformerBackend: m.TransformerBackend, projectionBackend: m.ProjectionBackend, sequenceBackend: m.SequenceBackend, output: output, fusedEOS: true, backbone: backbone, eosBackbone: eos, eosMask: append([]bool(nil), mask...), eosWeight: weight}, float64(out[1][0]), nil
}
func (b *ResidentTrainingBackend) evaluateForwardResults(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, weight float32) (*flowLMTrainingTape, float64, error) {
	if err := b.admitEOSBackbone(m); err != nil {
		return nil, 0, err
	}
	if !b.forwardResultsEnabled() || m.AttentionBackwardBackend != nil {
		return nil, 0, b.fail(fmt.Errorf("forward results owner/scope"))
	}
	if b.stagedBackboneEnabled(batch) {
		return b.evaluateStagedForward(m, batch, mask, weight)
	}
	p, err := m.buildEOSForwardTrainingTape(batch, mask, weight, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	p, err = buildForwardResultsTape(p, m, batch, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	outputs := forwardResultsOutputs(p)
	if b.eosAdmissionEnabled() {
		p, err = appendForwardEOSAdmission(p, m, batch, b.MaxTapeBytes)
		if err != nil {
			return nil, 0, b.fail(err)
		}
		outputs = append(outputs, p.Tape.Outputs[len(p.Tape.Outputs)-1])
	}
	var out [][]float32
	if b.flowHandoffEnabled() {
		if b.handoffLease != nil {
			return nil, 0, b.fail(fmt.Errorf("flow handoff producer already live"))
		}
		b.handoffRow = -1
		b.handoffLease, err = b.state.BeginZeroPrunedTapeLease(b.ctx, p.Tape, p.ResidentBindings, outputs, b.MaxTapeBytes)
		if err == nil {
			out, err = b.handoffLease.DownloadOutputs(b.ctx, outputs)
		}
		if err == nil {
			b.handoffHidden = out[0]
			b.handoffRoot, b.handoffRow = outputs[0], -1
			b.handoffFrames, b.handoffWidth = batch.Frames, m.Hidden
		}
	} else {
		out, err = b.runResidentTape(p.Tape, p.ResidentBindings, outputs, false, 1)
	}
	if err != nil {
		return nil, 0, b.fail(err)
	}
	if b.eosAdmissionEnabled() {
		if err := validateForwardEOSAdmission(out, batch); err != nil {
			return nil, 0, b.fail(err)
		}
		out = out[:3]
	}
	tape, loss, err := decodeForwardResults(out, m, batch, mask, weight, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	return tape, loss, nil
}
