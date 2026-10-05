package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type TrainingVoiceSequenceBackend interface {
	VoiceSequenceEnabled() bool
	EvaluateVoiceSequence(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, []float32, bool) (SequenceTrainingResult, error)
}

func (b *ResidentTrainingBackend) VoiceSequenceEnabled() bool {
	return b != nil && (b.ConditionerEnabled() || b.Scope == "ptx-resident-voicefused-hybrid-v1" || b.Scope == "vulkan-resident-voicefused-hybrid-v1")
}

// BuildVoiceSequenceTrainingTape projects raw voice rows within prefix/text/audio
// assembly and returns raw voice/audio VJPs. Empty voice still binds a one-row
// zero speaker input/seed so its parameter gradient is explicitly touched.
func (m *FlowLMTrainingCPU) BuildVoiceSequenceTrainingTape(batch FlowLMTrainingBatch, audio, seed []float32, maxBytes int64) (SequenceTrainingTape, error) {
	return m.buildVoiceSequenceTrainingTape(batch, audio, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildVoiceSequenceTrainingTape(batch FlowLMTrainingBatch, audio, seed []float32, maxBytes int64, borrowParameters bool) (SequenceTrainingTape, error) {
	if m == nil || m.Hidden < 1 || m.Hidden > 16384 || m.LatentDim < 1 || m.LatentDim > 16384 || m.Vocabulary < 1 || m.Vocabulary > 65536 || m.Vocabulary > (1<<26)/m.Hidden || batch.Frames < 1 || batch.Frames > 1024 || batch.VoiceFrames < 0 || batch.VoiceFrames > 1024 || len(batch.TextTokens) > 1024 {
		return SequenceTrainingTape{}, fmt.Errorf("voice sequence topology")
	}
	h, c := m.Hidden, m.LatentDim
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	rows := prefix + batch.Frames
	if rows > 1024 || m.SpeakerProjection.In != c || m.SpeakerProjection.Out != h || m.SpeakerProjection.Bias != nil || len(m.Embedding) != m.Vocabulary*h || len(m.BOSBeforeVoice) != h || len(batch.VoiceLatents) != batch.VoiceFrames*c || len(audio) != batch.Frames*h || len(seed) != rows*h || !finiteF32(batch.VoiceLatents) || !finiteF32(audio) || !finiteF32(seed) {
		return SequenceTrainingTape{}, fmt.Errorf("voice sequence lengths/values")
	}
	if err := validateOwnedTrainingLinear(m.SpeakerProjection, false); err != nil {
		return SequenceTrainingTape{}, err
	}
	for _, id := range batch.TextTokens {
		if uint64(id) >= uint64(m.Vocabulary) {
			return SequenceTrainingTape{}, fmt.Errorf("voice sequence token range")
		}
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	before := g.param("flow_lm.bos_before_voice", []int{h}, m.BOSBeforeVoice)
	table := g.param("flow_lm.conditioner.embed.weight", []int{m.Vocabulary * h}, m.Embedding)
	w := g.param("flow_lm.speaker_proj_weight", []int{h, c}, m.SpeakerProjection.Weight)
	voiceRows := max(1, batch.VoiceFrames)
	voice := g.zero(voiceRows, c)
	if batch.VoiceFrames > 0 {
		voice = g.constant([]int{voiceRows, c}, batch.VoiceLatents)
	}
	projected := g.command("linear-forward", []int{voice, g.zero(voiceRows, c), w, g.zero(h)}, []int{voiceRows, h}, 0)
	dy := g.constant([]int{rows * h}, seed)
	seq := g.pad(before, 0, rows*h)
	voiceSeed := g.zero(voiceRows, h)
	if batch.VoiceFrames > 0 {
		seq = g.sum(seq, g.pad(projected, h, rows*h), rows*h)
		voiceSeed = g.reshape(g.slice(dy, h, batch.VoiceFrames*h), voiceRows, h)
	}
	ids := make([]float32, max(1, len(batch.TextTokens)))
	for i, id := range batch.TextTokens {
		ids[i] = float32(id)
	}
	idTensor := g.constant([]int{len(ids)}, ids)
	textSeed := g.zero(h)
	if len(batch.TextTokens) > 0 {
		text := g.embedding(table, idTensor, len(ids), h, m.Vocabulary, 0)
		seq = g.sum(seq, g.pad(text, (1+batch.VoiceFrames)*h, rows*h), rows*h)
		textSeed = g.slice(dy, (1+batch.VoiceFrames)*h, len(batch.TextTokens)*h)
	}
	a := g.constant([]int{len(audio)}, audio)
	seq = g.sum(seq, g.pad(a, prefix*h, rows*h), rows*h)
	dtable := g.embedding(textSeed, idTensor, len(ids), h, m.Vocabulary, 1)
	dbefore := g.slice(dy, 0, h)
	operands := []int{voice, g.zero(voiceRows, c), w, voiceSeed, g.zero(voiceRows, h)}
	dvoice := g.command("linear-backward", operands, []int{voiceRows, c}, 0)
	dw := g.command("linear-backward", operands, []int{h, c}, 2)
	g.tape.Outputs = []int{seq}
	if batch.VoiceFrames > 0 {
		g.tape.Outputs = append(g.tape.Outputs, g.flat(dvoice))
	}
	g.tape.Outputs = append(g.tape.Outputs, g.slice(dy, prefix*h, len(audio)), dtable, dbefore, g.flat(dw))
	bindings := map[string]train.ResidentTapeBinding{"flow_lm.conditioner.embed.weight": {Parameter: table, Gradient: g.tape.Outputs[len(g.tape.Outputs)-3]}, "flow_lm.bos_before_voice": {Parameter: before, Gradient: dbefore}, "flow_lm.speaker_proj_weight": {Parameter: w, Gradient: g.tape.Outputs[len(g.tape.Outputs)-1]}}
	return materializeSequenceTape(g, bindings, maxBytes)
}
func (b *ResidentTrainingBackend) EvaluateVoiceSequence(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, audio, seed []float32, accumulate bool) (SequenceTrainingResult, error) {
	var r SequenceTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.VoiceSequenceEnabled() || b.trainer.FlowLM != m || m.SequenceBackend != b || m.ProjectionBackend != b {
		return r, b.fail(fmt.Errorf("voice sequence owner/scope"))
	}
	p, err := m.buildVoiceSequenceTrainingTape(batch, audio, seed, b.MaxTapeBytes, true)
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
	r = SequenceTrainingResult{Sequence: make([]float32, rows*m.Hidden), DVoice: make([]float32, batch.VoiceFrames*m.LatentDim), DAudio: make([]float32, len(audio))}
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
	if err = validateSequenceResult(r, rows*m.Hidden, batch.VoiceFrames*m.LatentDim, len(audio), batch.VoiceLatents, audio, seed); err != nil {
		return SequenceTrainingResult{}, b.fail(err)
	}
	return r, nil
}
