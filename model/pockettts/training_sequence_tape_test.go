package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func sequenceTinyModel() *FlowLMTrainingCPU {
	return &FlowLMTrainingCPU{Hidden: 2, Vocabulary: 3, LatentDim: 2, Embedding: []float32{1, 2, 3, 4, 5, 6}, BOSBeforeVoice: []float32{7, 8}, BOS: []float32{9, 10}}
}
func TestTrainingSequenceTapeReference(t *testing.T) {
	m := sequenceTinyModel()
	batch := FlowLMTrainingBatch{Frames: 2, VoiceFrames: 1, TextTokens: []uint32{2, 0, 2}}
	voice, audio := []float32{11, 12}, []float32{13, 14, 15, 16}
	seed := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	tape, err := m.BuildSequenceTrainingTape(batch, voice, audio, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := train.ValidateResidentTapeBindings(tape.Tape, tape.ResidentBindings, 1<<20); err != nil {
		t.Fatal(err)
	}
	got := interpretFlowTape(t, tape.Tape)
	assertSliceClose(t, "sequence", got[0], []float32{7, 8, 11, 12, 5, 6, 1, 2, 5, 6, 13, 14, 15, 16}, 0)
	assertSliceClose(t, "voice seed", got[1], []float32{3, 4}, 0)
	assertSliceClose(t, "audio seed", got[2], []float32{11, 12, 13, 14}, 0)
	assertSliceClose(t, "embedding repeated-token sum", got[3], []float32{7, 8, 0, 0, 14, 16}, 0)
	assertSliceClose(t, "before voice", got[4], []float32{1, 2}, 0)
	for _, frames := range []int{1, 3} {
		b := FlowLMTrainingBatch{Frames: frames, NormalizedLatents: []float32{1, 2, 3, 4, 5, 6}[:frames*2]}
		dy := []float32{11, 12, 13, 14, 15, 16}[:frames*2]
		tape, err := m.BuildAudioAssemblyTrainingTape(b, dy, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, tape.Tape)
		want := append([]float32{9, 10}, b.NormalizedLatents[:(frames-1)*2]...)
		assertSliceClose(t, "shift", got[0], want, 0)
		dn := make([]float32, frames*2)
		copy(dn, dy[2:])
		assertSliceClose(t, "unshift gradient", got[1], dn, 0)
		assertSliceClose(t, "BOS gradient", got[2], []float32{11, 12}, 0)
	}
	empty := FlowLMTrainingBatch{Frames: 1}
	tape, err = m.BuildSequenceTrainingTape(empty, nil, []float32{1, 2}, []float32{3, 4, 5, 6}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	got = interpretFlowTape(t, tape.Tape)
	assertSliceClose(t, "empty prefix", got[0], []float32{7, 8, 1, 2}, 0)
	assertSliceClose(t, "empty text table gradient", got[2], make([]float32, 6), 0)
}
func TestTrainingSequenceTapeAdmission(t *testing.T) {
	m := sequenceTinyModel()
	b := FlowLMTrainingBatch{Frames: 1, TextTokens: []uint32{0}}
	audio, seed := []float32{1, 2}, make([]float32, 6)
	for _, bad := range []FlowLMTrainingBatch{{Frames: 0}, {Frames: 1025}, {Frames: 1, VoiceFrames: -1}, {Frames: 1, TextTokens: []uint32{3}}} {
		if _, err := m.BuildSequenceTrainingTape(bad, nil, audio, seed, 1<<20); err == nil {
			t.Fatal("invalid batch")
		}
	}
	if _, err := m.BuildSequenceTrainingTape(b, nil, audio, seed, 1); err == nil {
		t.Fatal("ceiling")
	}
	if _, err := m.BuildSequenceTrainingTape(b, nil, []float32{float32(math.NaN()), 0}, seed, 1<<20); err == nil {
		t.Fatal("nonfinite")
	}
	if _, err := m.BuildAudioAssemblyTrainingTape(FlowLMTrainingBatch{Frames: 1}, audio, 1<<20); err == nil {
		t.Fatal("target length")
	}
	valid := SequenceTrainingResult{Sequence: []float32{1, 2}, DAudio: []float32{3, 4}}
	if validateSequenceResult(valid, 2, 0, 2) != nil {
		t.Fatal("boundary")
	}
	if validateSequenceResult(valid, 3, 0, 2) == nil {
		t.Fatal("bad length")
	}
	if validateSequenceResult(valid, 2, 0, 2, valid.Sequence) == nil {
		t.Fatal("borrowed boundary")
	}
	valid.DAudio = valid.Sequence
	if validateSequenceResult(valid, 2, 0, 2) == nil {
		t.Fatal("overlap")
	}
}
