package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type TrainingConditionerBackend interface {
	ConditionerEnabled() bool
	EvaluateConditioner(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, bool) (SequenceTrainingResult, error)
}

func (b *ResidentTrainingBackend) ConditionerEnabled() bool {
	return b != nil && (b.BackboneEnabled() || b.Scope == "ptx-resident-conditioner-hybrid-v1" || b.Scope == "vulkan-resident-conditioner-hybrid-v1")
}

// BuildConditionerTrainingTape combines raw voice/text and shifted audio
// projection/assembly VJPs. DAudio in the result denotes raw normalized-latent
// VJP, not projected-audio seed. All five parameter gradients accumulate once.
func (m *FlowLMTrainingCPU) BuildConditionerTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64) (SequenceTrainingTape, error) {
	return m.buildConditionerTrainingTape(batch, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildConditionerTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64, borrowParameters bool) (SequenceTrainingTape, error) {
	if err := m.validateConditionerTrainingTape(batch, seed); err != nil {
		return SequenceTrainingTape{}, err
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	cache := m.appendConditionerForwardGraph(g, batch)
	dy := g.constant([]int{len(seed)}, seed)
	outputs, bindings := m.appendConditionerBackwardGraph(g, batch, cache, dy)
	g.tape.Outputs = append([]int{cache.sequence}, outputs...)
	for _, name := range []string{"flow_lm.conditioner.embed.weight", "flow_lm.bos_before_voice", "flow_lm.speaker_proj_weight", "flow_lm.input_linear.weight", "flow_lm.bos_emb"} {
		g.tape.Outputs = append(g.tape.Outputs, bindings[name].Gradient)
	}
	return materializeSequenceTape(g, bindings, maxBytes)
}

func (m *FlowLMTrainingCPU) validateConditionerTrainingTape(batch FlowLMTrainingBatch, seed []float32) error {
	if m == nil || m.Hidden < 1 || m.Hidden > 16384 || m.LatentDim < 1 || m.LatentDim > 16384 || m.Vocabulary < 1 || m.Vocabulary > 65536 || m.Vocabulary > (1<<26)/m.Hidden || batch.Frames < 1 || batch.Frames > 1024 || batch.VoiceFrames < 0 || batch.VoiceFrames > 1024 || len(batch.TextTokens) > 1024 {
		return fmt.Errorf("voice sequence topology")
	}
	h, c := m.Hidden, m.LatentDim
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	rows := prefix + batch.Frames
	if rows > 1024 || m.SpeakerProjection.In != c || m.SpeakerProjection.Out != h || m.SpeakerProjection.Bias != nil || len(m.Embedding) != m.Vocabulary*h || len(m.BOSBeforeVoice) != h || len(batch.VoiceLatents) != batch.VoiceFrames*c || len(batch.NormalizedLatents) != batch.Frames*c || len(m.BOS) != c || m.Input.In != c || m.Input.Out != h || m.Input.Bias != nil || len(seed) != rows*h || !finiteF32(batch.VoiceLatents) || !finiteF32(batch.NormalizedLatents) || !finiteF32(m.BOS) || !finiteF32(seed) {
		return fmt.Errorf("voice sequence lengths/values")
	}
	if err := validateOwnedTrainingLinear(m.SpeakerProjection, false); err != nil {
		return err
	}
	if err := validateOwnedTrainingLinear(m.Input, false); err != nil {
		return err
	}
	for _, id := range batch.TextTokens {
		if uint64(id) >= uint64(m.Vocabulary) {
			return fmt.Errorf("voice sequence token range")
		}
	}
	return nil
}

type conditionerGraphCache struct{ sequence, before, table, w, voice, voiceRows, idTensor, idRows, bos, ax, aw int }

func (m *FlowLMTrainingCPU) appendConditionerForwardGraph(g *transformerTapeGraph, batch FlowLMTrainingBatch) conditionerGraphCache {
	h, c := m.Hidden, m.LatentDim
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	rows := prefix + batch.Frames
	before := g.param("flow_lm.bos_before_voice", []int{h}, m.BOSBeforeVoice)
	table := g.param("flow_lm.conditioner.embed.weight", []int{m.Vocabulary * h}, m.Embedding)
	w := g.param("flow_lm.speaker_proj_weight", []int{h, c}, m.SpeakerProjection.Weight)
	voiceRows := max(1, batch.VoiceFrames)
	voice := g.zero(voiceRows, c)
	if batch.VoiceFrames > 0 {
		voice = g.constant([]int{voiceRows, c}, batch.VoiceLatents)
	}
	projected := g.command("linear-forward", []int{voice, g.zero(voiceRows, c), w, g.zero(h)}, []int{voiceRows, h}, 0)
	seq := g.pad(before, 0, rows*h)
	if batch.VoiceFrames > 0 {
		seq = g.sum(seq, g.pad(projected, h, rows*h), rows*h)
	}
	ids := make([]float32, max(1, len(batch.TextTokens)))
	for i, id := range batch.TextTokens {
		ids[i] = float32(id)
	}
	idTensor := g.constant([]int{len(ids)}, ids)
	if len(batch.TextTokens) > 0 {
		text := g.embedding(table, idTensor, len(ids), h, m.Vocabulary, 0)
		seq = g.sum(seq, g.pad(text, (1+batch.VoiceFrames)*h, rows*h), rows*h)
	}
	bos := g.param("flow_lm.bos_emb", []int{c}, m.BOS)
	normalized := g.constant([]int{batch.Frames * c}, batch.NormalizedLatents)
	shifted := g.pad(bos, 0, batch.Frames*c)
	if batch.Frames > 1 {
		shifted = g.sum(shifted, g.pad(g.slice(normalized, 0, (batch.Frames-1)*c), c, batch.Frames*c), batch.Frames*c)
	}
	ax := g.reshape(shifted, batch.Frames, c)
	aw := g.param("flow_lm.input_linear.weight", []int{h, c}, m.Input.Weight)
	a := g.command("linear-forward", []int{ax, g.zero(batch.Frames, c), aw, g.zero(h)}, []int{batch.Frames, h}, 0)
	seq = g.sum(seq, g.pad(a, prefix*h, rows*h), rows*h)
	return conditionerGraphCache{seq, before, table, w, voice, voiceRows, idTensor, len(ids), bos, ax, aw}
}
func (m *FlowLMTrainingCPU) appendConditionerBackwardGraph(g *transformerTapeGraph, batch FlowLMTrainingBatch, cache conditionerGraphCache, dy int) ([]int, map[string]train.ResidentTapeBinding) {
	h, c := m.Hidden, m.LatentDim
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	before, table, w, voice, voiceRows, idTensor, bos, ax, aw := cache.before, cache.table, cache.w, cache.voice, cache.voiceRows, cache.idTensor, cache.bos, cache.ax, cache.aw
	voiceSeed := g.zero(voiceRows, h)
	if batch.VoiceFrames > 0 {
		voiceSeed = g.reshape(g.slice(dy, h, batch.VoiceFrames*h), voiceRows, h)
	}
	textSeed := g.zero(h)
	if len(batch.TextTokens) > 0 {
		textSeed = g.slice(dy, (1+batch.VoiceFrames)*h, len(batch.TextTokens)*h)
	}
	audioSeed := g.reshape(g.slice(dy, prefix*h, batch.Frames*h), batch.Frames, h)
	audioOperands := []int{ax, g.zero(batch.Frames, c), aw, audioSeed, g.zero(batch.Frames, h)}
	adx := g.command("linear-backward", audioOperands, []int{batch.Frames, c}, 0)
	adw := g.command("linear-backward", audioOperands, []int{h, c}, 2)
	dbos := g.slice(adx, 0, c)
	dn := g.zero(batch.Frames * c)
	if batch.Frames > 1 {
		dn = g.pad(g.slice(adx, c, (batch.Frames-1)*c), 0, batch.Frames*c)
	}
	dtable := g.embedding(textSeed, idTensor, cache.idRows, h, m.Vocabulary, 1)
	dbefore := g.slice(dy, 0, h)
	operands := []int{voice, g.zero(voiceRows, c), w, voiceSeed, g.zero(voiceRows, h)}
	dvoice := g.command("linear-backward", operands, []int{voiceRows, c}, 0)
	dw := g.command("linear-backward", operands, []int{h, c}, 2)
	outputs := []int{}
	if batch.VoiceFrames > 0 {
		outputs = append(outputs, g.flat(dvoice))
	}
	outputs = append(outputs, dn)
	bindings := map[string]train.ResidentTapeBinding{
		"flow_lm.conditioner.embed.weight": {Parameter: table, Gradient: dtable},
		"flow_lm.bos_before_voice":         {Parameter: before, Gradient: dbefore},
		"flow_lm.speaker_proj_weight":      {Parameter: w, Gradient: g.flat(dw)},
		"flow_lm.input_linear.weight":      {Parameter: aw, Gradient: g.flat(adw)},
		"flow_lm.bos_emb":                  {Parameter: bos, Gradient: dbos},
	}
	return outputs, bindings
}

func (b *ResidentTrainingBackend) EvaluateConditioner(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed []float32, accumulate bool) (SequenceTrainingResult, error) {
	var r SequenceTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.ConditionerEnabled() || b.trainer.FlowLM != m || m.SequenceBackend != b || m.ProjectionBackend != b {
		return r, b.fail(fmt.Errorf("conditioner owner/scope"))
	}
	p, err := m.buildConditionerTrainingTape(batch, seed, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	count := 2
	if batch.VoiceFrames > 0 {
		count = 3
	}
	outputs := p.Tape.Outputs[:count]
	forward := !accumulate && zeroTrainingSeed(seed)
	if accumulate {
		outputs = outputs[1:]
	} else if forward {
		outputs = outputs[:1]
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return r, b.fail(err)
	}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	r = SequenceTrainingResult{Sequence: make([]float32, rows*m.Hidden), DVoice: make([]float32, batch.VoiceFrames*m.LatentDim), DAudio: make([]float32, batch.Frames*m.LatentDim)}
	index := 0
	if !accumulate {
		r.Sequence = out[0]
		index = 1
	}
	if !forward {
		if batch.VoiceFrames > 0 {
			r.DVoice = out[index]
			index++
		}
		r.DAudio = out[index]
	}
	if err = validateSequenceResult(r, rows*m.Hidden, batch.VoiceFrames*m.LatentDim, batch.Frames*m.LatentDim, batch.VoiceLatents, batch.NormalizedLatents, seed); err != nil {
		return SequenceTrainingResult{}, b.fail(err)
	}
	for _, v := range r.DAudio[(batch.Frames-1)*m.LatentDim:] {
		if v != 0 {
			return SequenceTrainingResult{}, b.fail(fmt.Errorf("conditioner final normalized VJP"))
		}
	}
	return r, nil
}
