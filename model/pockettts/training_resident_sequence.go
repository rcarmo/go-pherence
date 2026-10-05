package pockettts

import (
	"fmt"
)

func (b *ResidentTrainingBackend) ownsTime() bool {
	return b != nil && (b.Scope == "ptx-resident-time-hybrid-v1" || b.Scope == "vulkan-resident-time-hybrid-v1" || b.ownsEOS())
}
func (b *ResidentTrainingBackend) ownsSequence() bool {
	return b != nil && (b.Scope == "ptx-resident-sequence-hybrid-v1" || b.Scope == "vulkan-resident-sequence-hybrid-v1" || b.ownsTime())
}
func (b *ResidentTrainingBackend) sequenceOwnerMatches(t *FullTrainer) bool {
	if b.ownsSequence() {
		return t.FlowLM.SequenceBackend == b
	}
	return t.FlowLM.SequenceBackend == nil
}
func sequenceParameter(name string) bool {
	return name == "flow_lm.conditioner.embed.weight" || name == "flow_lm.bos_emb" || name == "flow_lm.bos_before_voice"
}
func (b *ResidentTrainingBackend) EvaluateAudioAssembly(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, seed []float32, accumulate bool) (AudioAssemblyTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return AudioAssemblyTrainingResult{}, err
	}
	if !b.ownsSequence() || m != b.trainer.FlowLM {
		return AudioAssemblyTrainingResult{}, b.fail(fmt.Errorf("resident audio assembly owner/scope changed"))
	}
	tape, err := m.buildAudioAssemblyTrainingTape(batch, seed, b.MaxTapeBytes, true)
	if err != nil {
		return AudioAssemblyTrainingResult{}, b.fail(err)
	}
	outputs := tape.Tape.Outputs[:2]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(seed)
	if forward {
		outputs = outputs[:1]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return AudioAssemblyTrainingResult{}, b.fail(err)
	}
	if forward {
		values = residentForwardBoundary(values, []int{len(seed), len(seed)})
	}
	return AudioAssemblyTrainingResult{AudioInput: values[0], DNormalized: values[1]}, nil
}
func (b *ResidentTrainingBackend) EvaluateSequence(m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, voice, audio, seed []float32, accumulate bool) (SequenceTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return SequenceTrainingResult{}, err
	}
	if !b.ownsSequence() || m != b.trainer.FlowLM {
		return SequenceTrainingResult{}, b.fail(fmt.Errorf("resident sequence owner/scope changed"))
	}
	tape, err := m.buildSequenceTrainingTape(batch, voice, audio, seed, b.MaxTapeBytes, true)
	if err != nil {
		return SequenceTrainingResult{}, b.fail(err)
	}
	boundary := 2
	if batch.VoiceFrames > 0 {
		boundary = 3
	}
	outputs := tape.Tape.Outputs[:boundary]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(seed)
	if forward {
		outputs = outputs[:1]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return SequenceTrainingResult{}, b.fail(err)
	}
	if forward {
		lengths := []int{len(seed)}
		if batch.VoiceFrames > 0 {
			lengths = append(lengths, len(voice))
		}
		lengths = append(lengths, len(audio))
		values = residentForwardBoundary(values, lengths)
	}
	r := SequenceTrainingResult{Sequence: values[0], DAudio: values[len(values)-1]}
	if batch.VoiceFrames > 0 {
		r.DVoice = values[1]
	}
	return r, nil
}
func validateAudioAssemblyResult(r AudioAssemblyTrainingResult, n int, inputs ...[]float32) error {
	return validateProjectionResult(ProjectionTrainingResult{Output: r.AudioInput, DInput: r.DNormalized}, 1, n, n, inputs...)
}
func validateSequenceResult(r SequenceTrainingResult, n, v, a int, inputs ...[]float32) error {
	if len(r.Sequence) != n || len(r.DVoice) != v || len(r.DAudio) != a || !finiteF32(r.Sequence) || !finiteF32(r.DVoice) || !finiteF32(r.DAudio) {
		return fmt.Errorf("sequence boundary shape/nonfinite")
	}
	outputs := [][]float32{r.Sequence, r.DVoice, r.DAudio}
	for i, x := range outputs {
		for _, y := range outputs[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("sequence outputs overlap")
			}
		}
		for _, y := range inputs {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("sequence borrowed input")
			}
		}
	}
	return nil
}
