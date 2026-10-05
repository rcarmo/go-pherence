package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Sequence tapes use device gather, sequence-order embedding VJP and disjoint
// slices/pads. They retain CPU-facing boundaries; not a fused resident graph.
type SequenceTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}
type SequenceTrainingResult struct{ Sequence, DVoice, DAudio []float32 }
type AudioAssemblyTrainingResult struct{ AudioInput, DNormalized []float32 }
type TrainingSequenceBackend interface {
	EvaluateAudioAssembly(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, bool) (AudioAssemblyTrainingResult, error)
	EvaluateSequence(*FlowLMTrainingCPU, FlowLMTrainingBatch, []float32, []float32, []float32, bool) (SequenceTrainingResult, error)
}

func (g *transformerTapeGraph) slice(x, offset, n int) int {
	id := g.command("vector", []int{g.flat(x), g.zero(1)}, []int{n}, 1)
	g.tape.Commands[len(g.tape.Commands)-1].Offset = offset
	return id
}
func (g *transformerTapeGraph) pad(x, offset, n int) int {
	id := g.command("vector", []int{g.flat(x), g.zero(1)}, []int{n}, 2)
	g.tape.Commands[len(g.tape.Commands)-1].Offset = offset
	return id
}
func (g *transformerTapeGraph) embedding(x, ids, rows, width, vocabulary int, mode uint32) int {
	_, n, _ := train.EmbeddingShape(rows, width, vocabulary, mode)
	id := g.command("embedding", []int{g.flat(x), ids}, []int{n}, mode)
	c := &g.tape.Commands[len(g.tape.Commands)-1]
	c.Rows, c.Width, c.Parts = rows, width, vocabulary
	return id
}
func materializeSequenceTape(g *transformerTapeGraph, bindings map[string]train.ResidentTapeBinding, maxBytes int64) (SequenceTrainingTape, error) {
	var result SequenceTrainingTape
	if maxBytes <= 0 {
		return result, fmt.Errorf("sequence tape ceiling required")
	}
	var bytes int64
	for id, s := range g.tape.Tensors {
		n, err := s.Elements()
		if err != nil {
			return result, err
		}
		bytes += int64(n) * 4
		if v, ok := g.constants[id]; ok && (len(v) != n || !finiteF32(v)) {
			return result, fmt.Errorf("sequence tape constant shape/nonfinite")
		}
	}
	if bytes > maxBytes {
		var views, zeros, constants int64
		for id, s := range g.tape.Tensors {
			n, _ := s.Elements()
			if g.zeros[id] {
				zeros += int64(n) * 4
			} else if _, ok := g.constants[id]; ok {
				constants += int64(n) * 4
			}
		}
		for _, c := range g.tape.Commands {
			if c.Op == "reshape" {
				n, _ := g.tape.Tensors[c.Output].Elements()
				views += int64(n) * 4
			}
		}
		return result, fmt.Errorf("sequence tape byte ceiling: complete source=%d cap=%d views=%d zeros=%d constants=%d", bytes, maxBytes, views, zeros, constants)
	}
	for id, s := range g.tape.Tensors {
		if v, ok := g.constants[id]; ok {
			g.tape.Tensors[id].Values = g.constantValues(id, v)
		} else if g.zeros[id] {
			n, _ := s.Elements()
			g.tape.Tensors[id].Values = g.zeroValues(n)
		}
	}
	if err := g.tape.Validate(maxBytes); err != nil {
		return result, err
	}
	result.Tape, result.ResidentBindings = g.tape, bindings
	return result, nil
}
func (m *FlowLMTrainingCPU) BuildAudioAssemblyTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64) (SequenceTrainingTape, error) {
	return m.buildAudioAssemblyTrainingTape(batch, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildAudioAssemblyTrainingTape(batch FlowLMTrainingBatch, seed []float32, maxBytes int64, borrowParameters bool) (SequenceTrainingTape, error) {
	if m == nil || m.LatentDim < 1 || m.LatentDim > 16384 || batch.Frames < 1 || batch.Frames > 1024 || len(m.BOS) != m.LatentDim || len(batch.NormalizedLatents) != batch.Frames*m.LatentDim || len(seed) != len(batch.NormalizedLatents) {
		return SequenceTrainingTape{}, fmt.Errorf("audio assembly topology")
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	c, n := m.LatentDim, len(seed)
	bos := g.param("flow_lm.bos_emb", []int{c}, m.BOS)
	x := g.constant([]int{n}, batch.NormalizedLatents)
	dy := g.constant([]int{n}, seed)
	audio := g.pad(bos, 0, n)
	dn := g.zero(n)
	if batch.Frames > 1 {
		audio = g.sum(audio, g.pad(g.slice(x, 0, n-c), c, n), n)
		dn = g.pad(g.slice(dy, c, n-c), 0, n)
	}
	dbos := g.slice(dy, 0, c)
	g.tape.Outputs = []int{audio, dn, dbos}
	return materializeSequenceTape(g, map[string]train.ResidentTapeBinding{"flow_lm.bos_emb": {Parameter: bos, Gradient: dbos}}, maxBytes)
}
func (m *FlowLMTrainingCPU) BuildSequenceTrainingTape(batch FlowLMTrainingBatch, voice, audio, seed []float32, maxBytes int64) (SequenceTrainingTape, error) {
	return m.buildSequenceTrainingTape(batch, voice, audio, seed, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildSequenceTrainingTape(batch FlowLMTrainingBatch, voice, audio, seed []float32, maxBytes int64, borrowParameters bool) (SequenceTrainingTape, error) {
	if m == nil || m.Hidden < 1 || m.Hidden > 16384 || m.Vocabulary < 1 || m.Vocabulary > 65536 || m.Vocabulary > (1<<26)/m.Hidden || batch.Frames < 1 || batch.Frames > 1024 || batch.VoiceFrames < 0 || batch.VoiceFrames > 1024 || len(batch.TextTokens) > 1024 {
		return SequenceTrainingTape{}, fmt.Errorf("sequence assembly topology")
	}
	h := m.Hidden
	prefix := 1 + batch.VoiceFrames + len(batch.TextTokens)
	rows := prefix + batch.Frames
	if rows > 1024 || len(m.Embedding) != m.Vocabulary*h || len(m.BOSBeforeVoice) != h || len(voice) != batch.VoiceFrames*h || len(audio) != batch.Frames*h || len(seed) != rows*h {
		return SequenceTrainingTape{}, fmt.Errorf("sequence assembly lengths")
	}
	for _, token := range batch.TextTokens {
		if uint64(token) >= uint64(m.Vocabulary) {
			return SequenceTrainingTape{}, fmt.Errorf("sequence token out of range")
		}
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	before := g.param("flow_lm.bos_before_voice", []int{h}, m.BOSBeforeVoice)
	table := g.param("flow_lm.conditioner.embed.weight", []int{m.Vocabulary * h}, m.Embedding)
	dy := g.constant([]int{rows * h}, seed)
	seq := g.pad(before, 0, rows*h)
	if len(voice) > 0 {
		v := g.constant([]int{len(voice)}, voice)
		seq = g.sum(seq, g.pad(v, h, rows*h), rows*h)
	}
	ids := make([]float32, max(1, len(batch.TextTokens)))
	for i, token := range batch.TextTokens {
		ids[i] = float32(token)
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
	g.tape.Outputs = []int{seq}
	if len(voice) > 0 {
		g.tape.Outputs = append(g.tape.Outputs, g.slice(dy, h, len(voice)))
	}
	g.tape.Outputs = append(g.tape.Outputs, g.slice(dy, prefix*h, len(audio)), dtable, dbefore)
	bindings := map[string]train.ResidentTapeBinding{"flow_lm.conditioner.embed.weight": {Parameter: table, Gradient: dtable}, "flow_lm.bos_before_voice": {Parameter: before, Gradient: dbefore}}
	return materializeSequenceTape(g, bindings, maxBytes)
}
