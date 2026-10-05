package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

func (b *ResidentTrainingBackend) eosForwardEnabled() bool {
	return b != nil && (b.forwardResultsEnabled() || b.Scope == "ptx-resident-eosforward-hybrid-v1" || b.Scope == "vulkan-resident-eosforward-hybrid-v1")
}

// Private step-only read-only forward. Full backward bindings remain admitted;
// pruning retains sequence/transformed rows and unweighted EOS mean only.
// Topology/parameters/mask/weight are admitted before dispatch with zero hidden
// placeholders. The actual hidden absolute-dot bound is checked after download
// and before any backward/parameter accumulation (no state mutation here).
func (m *FlowLMTrainingCPU) buildEOSForwardTrainingTape(batch FlowLMTrainingBatch, mask []bool, weight float32, maxBytes int64) (BackboneTrainingTape, error) {
	if m == nil || batch.Frames < 1 || batch.Frames > 1024 || m.Hidden < 1 || m.Hidden > 16384 || batch.VoiceFrames < 0 || batch.VoiceFrames > 1024 || len(batch.TextTokens) > 1024 {
		return BackboneTrainingTape{}, fmt.Errorf("EOS forward topology")
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	seed := make([]float32, rows*m.Hidden)
	if _, _, err := m.validateMode(batch, make([]float32, batch.Frames*m.Hidden), make([]float32, batch.Frames), false); err != nil {
		return BackboneTrainingTape{}, err
	}
	if owner, ok := m.EOSBackend.(*ResidentTrainingBackend); ok && owner.sharedWorkspaceEnabled() {
		return m.buildSharedEOSForwardOnlyTape(batch, seed, mask, weight, maxBytes)
	}
	p, err := m.buildEOSBackboneTrainingTape(batch, seed, make([]float32, batch.Frames*m.Hidden), mask, weight, maxBytes, true)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	// Locate the admitted EOS operation, retaining the exact affine path shared
	// with the existing backward graph. Do not infer IDs from tensor shapes.
	var eos train.Command
	found := false
	for _, c := range p.Tape.Commands {
		if c.Op == "eos" && c.Mode == 1 {
			eos = c
			found = true
			break
		}
	}
	if !found {
		return BackboneTrainingTape{}, fmt.Errorf("EOS forward missing loss edge")
	}
	n := len(p.Tape.Tensors)
	loss, mean := n, n+1
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{batch.Frames}}, train.TensorSpec{Shape: []int{1}})
	p.Tape.Commands = append(p.Tape.Commands, train.Command{Op: "eos", Inputs: append([]int(nil), eos.Inputs...), Output: loss, Mode: 0, Rows: eos.Rows, Scale: weight}, train.Command{Op: "eos", Inputs: []int{loss, eos.Inputs[1]}, Output: mean, Mode: 2, Rows: eos.Rows, Scale: weight})
	// Keep all bindings declared, even though this call never accumulates them.
	p.Tape.Outputs = append(p.Tape.Outputs, mean)
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return BackboneTrainingTape{}, err
	}
	return p, nil
}

// The shared producer declares only forward operations. Parameter placeholders
// remain a complete named inventory, admitted even though no gradient executes.
func (m *FlowLMTrainingCPU) buildSharedEOSForwardOnlyTape(batch FlowLMTrainingBatch, seed []float32, mask []bool, weight float32, maxBytes int64) (BackboneTrainingTape, error) {
	if err := m.validateConditionerTrainingTape(batch, seed); err != nil {
		return BackboneTrainingTape{}, err
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	if err := validateTrainableTransformer(m.Transformer, seed, seed, rows); err != nil {
		return BackboneTrainingTape{}, err
	}
	if err := preflightTransformerTrainingTape(m.Transformer, rows); err != nil {
		return BackboneTrainingTape{}, err
	}
	_, _, err := m.admitEOSProjection(make([]float32, batch.Frames*m.Hidden), mask, weight, maxBytes)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	var placeholderPool *trainingZeroPool
	if owner, ok := m.EOSBackend.(*ResidentTrainingBackend); ok && owner.forwardZeroWorkspace != nil {
		ws := owner.forwardZeroWorkspace
		if ws.zeroGradientOwner != owner || ws.frames != batch.Frames || ws.zeroGradientPool == nil || !ws.compatible(m, ws.flow) {
			return BackboneTrainingTape{}, fmt.Errorf("forward zero workspace provenance")
		}
		placeholderPool = ws.zeroGradientPool
	}
	g := &transformerTapeGraph{borrowParameters: true, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, forwardOnly: true}
	cache := m.appendConditionerForwardGraph(g, batch)
	output, _, _, err := m.Transformer.appendTransformerTrainingGraphSeed(g, g.reshape(cache.sequence, rows, m.Hidden), 0, rows, nil)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	hidden := g.reshape(g.slice(g.flat(output), (rows-batch.Frames)*m.Hidden, batch.Frames*m.Hidden), batch.Frames, m.Hidden)
	ew := g.param("flow_lm.out_eos.weight", []int{1, m.Hidden}, m.EOS.Weight)
	eb := g.param("flow_lm.out_eos.bias", []int{1}, m.EOS.Bias)
	logits := g.command("linear-forward", []int{hidden, g.zero(batch.Frames, m.Hidden), ew, eb}, []int{batch.Frames, 1}, 0)
	labels, active, err := eosTrainingLabels(mask)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	mi := g.constant([]int{batch.Frames}, labels)
	loss := g.command("eos", []int{g.flat(logits), mi}, []int{batch.Frames}, 0)
	g.tape.Commands[len(g.tape.Commands)-1].Rows = active
	g.tape.Commands[len(g.tape.Commands)-1].Scale = weight
	mean := g.command("eos", []int{loss, mi}, []int{1}, 2)
	g.tape.Commands[len(g.tape.Commands)-1].Rows = active
	g.tape.Commands[len(g.tape.Commands)-1].Scale = weight
	bindings := map[string]train.ResidentTapeBinding{}
	g.tape.Outputs = []int{cache.sequence, output}
	names := make([]string, 0, len(g.params))
	for name := range g.params {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		id := g.params[name]
		n, _ := g.tape.Tensors[id].Elements()
		gradient := g.uniqueZero(n)
		if placeholderPool != nil {
			if values, ok := (*placeholderPool)[n]; ok {
				// The extent was admitted when the workspace was constructed.
				// Never grow its retained pool with graph-specific scratch shapes.
				g.constants[gradient] = values
				delete(g.zeros, gradient)
				if g.borrowed == nil {
					g.borrowed = map[int]bool{}
				}
				g.borrowed[gradient] = true
			}
		}
		bindings[name] = train.ResidentTapeBinding{Parameter: id, Gradient: gradient}
		g.tape.Outputs = append(g.tape.Outputs, gradient)
	}
	g.tape.Outputs = append(g.tape.Outputs, mean)
	built, err := materializeSequenceTape(g, bindings, maxBytes)
	if err != nil {
		return BackboneTrainingTape{}, err
	}
	if err = train.ValidateResidentTapeBindings(built.Tape, bindings, maxBytes); err != nil {
		return BackboneTrainingTape{}, err
	}
	return BackboneTrainingTape{Tape: built.Tape, ResidentBindings: bindings}, nil
}

func decodeEOSForward(out [][]float32, m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, weight float32, maxBytes int64) (*flowLMTrainingTape, float64, error) {
	if len(out) != 3 {
		return nil, 0, fmt.Errorf("EOS forward boundary count")
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	prefix := rows - batch.Frames
	values := out
	sizes := []int{rows * m.Hidden, rows * m.Hidden, 1}
	for i, x := range values {
		if len(x) != sizes[i] || !finiteF32(x) {
			return nil, 0, fmt.Errorf("EOS forward shape/nonfinite")
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return nil, 0, fmt.Errorf("EOS forward aliased outputs")
			}
		}
		for _, source := range [][]float32{batch.NormalizedLatents, batch.VoiceLatents} {
			if f32SlicesOverlap(x, source) {
				return nil, 0, fmt.Errorf("EOS forward borrowed input")
			}
		}
	}
	if out[2][0] < 0 {
		return nil, 0, fmt.Errorf("EOS forward negative loss")
	}
	hidden := out[1][prefix*m.Hidden:]
	// Same bound as the separate forward EOS path. Reject even finite logits
	// from cancellation when the absolute-dot sum exceeds admitted headroom.
	if _, _, err := m.admitEOSProjection(hidden, mask, weight, maxBytes); err != nil {
		return nil, 0, err
	}
	output := FlowLMTrainingOutput{Z: append([]float32(nil), hidden...), EOS: make([]float32, batch.Frames), Sequence: out[0], PrefixRows: prefix, SequenceRows: rows}
	backend, ok := m.SequenceBackend.(TrainingBackboneBackend)
	if !ok {
		return nil, 0, fmt.Errorf("EOS forward missing backbone")
	}
	eos, ok := m.EOSBackend.(TrainingEOSBackboneBackend)
	if !ok {
		return nil, 0, fmt.Errorf("EOS forward missing EOS owner")
	}
	return &flowLMTrainingTape{rows: rows, prefix: prefix, textOffset: 1 + batch.VoiceFrames, sequence: out[0], transformerBackend: m.TransformerBackend, projectionBackend: m.ProjectionBackend, sequenceBackend: m.SequenceBackend, output: output, fusedEOS: true, backbone: backend, eosBackbone: eos, eosMask: append([]bool(nil), mask...), eosWeight: weight}, float64(out[2][0]), nil
}
func (b *ResidentTrainingBackend) evaluateEOSForward(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, weight float32) (*flowLMTrainingTape, float64, error) {
	if err := b.admitEOSBackbone(m); err != nil {
		return nil, 0, err
	}
	if !b.eosForwardEnabled() || m.AttentionBackwardBackend != nil {
		return nil, 0, b.fail(fmt.Errorf("EOS forward owner/scope"))
	}
	p, err := m.buildEOSForwardTrainingTape(batch, mask, weight, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, []int{p.Tape.Outputs[0], p.Tape.Outputs[1], p.Tape.Outputs[len(p.Tape.Outputs)-1]}, false, 1)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	tape, loss, err := decodeEOSForward(out, m, batch, mask, weight, b.MaxTapeBytes)
	if err != nil {
		return nil, 0, b.fail(err)
	}
	return tape, loss, nil
}
