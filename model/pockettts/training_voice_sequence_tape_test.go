package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingVoiceSequenceTapeReference(t *testing.T) {
	for _, v := range []int{0, 1, 3} {
		for _, tokens := range [][]uint32{nil, {1, 1, 2}} {
			m := &FlowLMTrainingCPU{Hidden: 3, Vocabulary: 3, LatentDim: 2, Embedding: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, BOSBeforeVoice: []float32{.1, .2, .3}}
			m.LatentDim = 2
			m.SpeakerProjection = LinearF32{In: 2, Out: m.Hidden, Weight: []float32{.4, -.2, .7, .1, -.3, .6}}
			batch := FlowLMTrainingBatch{Frames: 2, VoiceFrames: v, VoiceLatents: make([]float32, v*2), TextTokens: tokens}
			for i := range batch.VoiceLatents {
				batch.VoiceLatents[i] = float32(i%7-3) * .1
			}
			rows := 1 + v + len(tokens) + 2
			audio := []float32{.2, .3, .4, .5, .6, .7}
			seed := make([]float32, rows*m.Hidden)
			for i := range seed {
				seed[i] = float32(i%11-5) * .03
			}
			p, err := m.BuildVoiceSequenceTrainingTape(batch, audio, seed, 1<<20)
			if err != nil {
				t.Fatal(v, tokens, err)
			}
			if err = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 1<<20); err != nil {
				t.Fatal(err)
			}
			out := interpretFlowTape(t, p.Tape)
			want := make([]float64, rows*m.Hidden)
			dx := make([]float64, v*2)
			dw := make([]float64, 6)
			dtable := make([]float64, len(m.Embedding))
			for k, x := range m.BOSBeforeVoice {
				want[k] = float64(x)
			}
			for row := 0; row < v; row++ {
				for o := 0; o < m.Hidden; o++ {
					for k := 0; k < 2; k++ {
						w := float64(m.SpeakerProjection.Weight[o*2+k])
						x := float64(batch.VoiceLatents[row*2+k])
						want[(row+1)*m.Hidden+o] += w * x
						dy := float64(seed[(row+1)*m.Hidden+o])
						dx[row*2+k] += dy * w
						dw[o*2+k] += dy * x
					}
				}
			}
			for row, id := range tokens {
				for k := 0; k < m.Hidden; k++ {
					index := (1+v+row)*m.Hidden + k
					want[index] = float64(m.Embedding[int(id)*m.Hidden+k])
					dtable[int(id)*m.Hidden+k] += float64(seed[index])
				}
			}
			prefix := 1 + v + len(tokens)
			for i, x := range audio {
				want[prefix*m.Hidden+i] = float64(x)
			}
			check := func(name string, a []float32, b []float64) {
				t.Helper()
				if len(a) != len(b) {
					t.Fatal(name, "shape")
				}
				for i, x := range a {
					if math.Abs(float64(x)-b[i]) > 2e-7+2e-6*math.Abs(b[i]) {
						t.Fatal(v, tokens, name, i, x, b[i])
					}
				}
			}
			check("sequence", out[0], want)
			offset := 1
			if v > 0 {
				check("raw voice VJP", out[1], dx)
				offset++
			}
			assertSliceClose(t, "audio VJP", out[offset], seed[prefix*m.Hidden:], 0)
			check("repeated-token VJP", out[offset+1], dtable)
			assertSliceClose(t, "BOSbefore VJP", out[offset+2], seed[:m.Hidden], 0)
			check("speaker VJP", out[offset+3], dw)
		}
	}
}
func TestTrainingVoiceSequenceTapeAdmission(t *testing.T) {
	m := &FlowLMTrainingCPU{Hidden: 3, Vocabulary: 3, LatentDim: 2, Embedding: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, BOSBeforeVoice: []float32{.1, .2, .3}}
	m.LatentDim = 2
	m.SpeakerProjection = LinearF32{In: 2, Out: 3, Weight: []float32{.4, -.2, .7, .1, -.3, .6}}
	batch := FlowLMTrainingBatch{Frames: 1, VoiceFrames: 1, VoiceLatents: []float32{.1, .2}, TextTokens: []uint32{1}}
	audio := []float32{.2, .3, .4}
	seed := make([]float32, 12)
	if _, err := m.BuildVoiceSequenceTrainingTape(batch, audio, seed, 1); err == nil {
		t.Fatal("budget")
	}
	bad := batch
	bad.TextTokens = []uint32{99}
	if _, err := m.BuildVoiceSequenceTrainingTape(bad, audio, seed, 1<<20); err == nil {
		t.Fatal("token")
	}
	bad = batch
	bad.VoiceLatents = []float32{float32(math.NaN()), .2}
	if _, err := m.BuildVoiceSequenceTrainingTape(bad, audio, seed, 1<<20); err == nil {
		t.Fatal("nonfinite")
	}
	if _, err := m.BuildVoiceSequenceTrainingTape(batch, audio, nil, 1<<20); err == nil {
		t.Fatal("seed")
	}
	m.SpeakerProjection.Bias = []float32{0, 0, 0}
	if _, err := m.BuildVoiceSequenceTrainingTape(batch, audio, seed, 1<<20); err == nil {
		t.Fatal("bias")
	}
	m.SpeakerProjection.Bias = nil
	p, err := m.BuildVoiceSequenceTrainingTape(batch, audio, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	before := interpretFlowTape(t, p.Tape)
	m.SpeakerProjection.Weight[0] = 99
	batch.VoiceLatents[0] = 99
	audio[0] = 99
	seed[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
}
