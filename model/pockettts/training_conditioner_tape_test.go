package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func conditionerFixture() *FlowLMTrainingCPU {
	return &FlowLMTrainingCPU{Hidden: 3, LatentDim: 2, Vocabulary: 3, BOS: []float32{.3, -.7}, BOSBeforeVoice: []float32{.1, .2, .3}, Embedding: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, Input: LinearF32{In: 2, Out: 3, Weight: []float32{.4, -.2, .7, .1, -.3, .6}}, SpeakerProjection: LinearF32{In: 2, Out: 3, Weight: []float32{.5, .2, -.1, .6, .3, -.4}}}
}
func TestTrainingConditionerTapeReference(t *testing.T) {
	for _, frames := range []int{1, 2, 7} {
		for _, voiceRows := range []int{0, 1, 3} {
			for _, tokens := range [][]uint32{nil, {1, 1, 2}} {
				m := conditionerFixture()
				batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: voiceRows, NormalizedLatents: make([]float32, frames*2), VoiceLatents: make([]float32, voiceRows*2), TextTokens: tokens}
				for i := range batch.NormalizedLatents {
					batch.NormalizedLatents[i] = float32(i%11-5) * .08
				}
				for i := range batch.VoiceLatents {
					batch.VoiceLatents[i] = float32(i%7-3) * .1
				}
				rows := 1 + voiceRows + len(tokens) + frames
				seed := make([]float32, rows*3)
				for i := range seed {
					seed[i] = float32(i%11-5) * .03
				}
				p, err := m.BuildConditionerTrainingTape(batch, seed, 1<<20)
				if err != nil {
					t.Fatal(frames, voiceRows, tokens, err)
				}
				if err = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 1<<20); err != nil {
					t.Fatal(err)
				}
				out := interpretFlowTape(t, p.Tape)
				want := make([]float64, rows*3)
				dv, dn := make([]float64, voiceRows*2), make([]float64, frames*2)
				sw, aw := make([]float64, 6), make([]float64, 6)
				bos := make([]float64, 2)
				emb := make([]float64, 9)
				for i, x := range m.BOSBeforeVoice {
					want[i] = float64(x)
				}
				for row := 0; row < voiceRows; row++ {
					for o := 0; o < 3; o++ {
						for k := 0; k < 2; k++ {
							dy := float64(seed[(row+1)*3+o])
							x := float64(batch.VoiceLatents[row*2+k])
							w := float64(m.SpeakerProjection.Weight[o*2+k])
							want[(row+1)*3+o] += x * w
							dv[row*2+k] += dy * w
							sw[o*2+k] += dy * x
						}
					}
				}
				for row, id := range tokens {
					for k := 0; k < 3; k++ {
						offset := (1+voiceRows+row)*3 + k
						want[offset] = float64(m.Embedding[int(id)*3+k])
						emb[int(id)*3+k] += float64(seed[offset])
					}
				}
				prefix := 1 + voiceRows + len(tokens)
				for row := 0; row < frames; row++ {
					x := m.BOS
					if row > 0 {
						x = batch.NormalizedLatents[(row-1)*2 : row*2]
					}
					for o := 0; o < 3; o++ {
						for k := 0; k < 2; k++ {
							dy := float64(seed[(prefix+row)*3+o])
							w := float64(m.Input.Weight[o*2+k])
							want[(prefix+row)*3+o] += float64(x[k]) * w
							aw[o*2+k] += dy * float64(x[k])
							if row == 0 {
								bos[k] += dy * w
							} else {
								dn[(row-1)*2+k] += dy * w
							}
						}
					}
				}
				check := func(name string, x []float32, y []float64) {
					t.Helper()
					if len(x) != len(y) {
						t.Fatal(name, "shape")
					}
					for i, v := range x {
						if math.Abs(float64(v)-y[i]) > 2e-7+2e-6*math.Abs(y[i]) {
							t.Fatal(frames, voiceRows, name, i, v, y[i])
						}
					}
				}
				check("sequence", out[0], want)
				index := 1
				if voiceRows > 0 {
					check("rawvoice VJP", out[index], dv)
					index++
				}
				check("normalized VJP", out[index], dn)
				check("embedding", out[index+1], emb)
				assertSliceClose(t, "BOS before", out[index+2], seed[:3], 0)
				check("speaker weight", out[index+3], sw)
				check("audio weight", out[index+4], aw)
				check("audio BOS", out[index+5], bos)
				// Independently compose previous bound audio and speaker/sequence tapes.
				af, err := m.BuildAudioProjectionTrainingTape(batch, make([]float32, frames*3), 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				av := interpretFlowTape(t, af.Tape)
				vs, err := m.BuildVoiceSequenceTrainingTape(batch, av[0], seed, 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				vv := interpretFlowTape(t, vs.Tape)
				ab, err := m.BuildAudioProjectionTrainingTape(batch, vv[index], 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				bv := interpretFlowTape(t, ab.Tape)
				assertSliceClose(t, "composed sequence", out[0], vv[0], 2e-7)
				if voiceRows > 0 {
					assertSliceClose(t, "composed voice", out[1], vv[1], 2e-7)
				}
				assertSliceClose(t, "composed normalized", out[index], bv[1], 2e-7)
				assertSliceClose(t, "composed embedding", out[index+1], vv[index+1], 2e-7)
				assertSliceClose(t, "composed speaker", out[index+3], vv[index+3], 2e-7)
				assertSliceClose(t, "composed audio weight", out[index+4], bv[2], 2e-7)
				assertSliceClose(t, "composed BOS", out[index+5], bv[3], 2e-7)
			}
		}
	}
}
func TestTrainingConditionerTapeAdmission(t *testing.T) {
	m := conditionerFixture()
	batch := FlowLMTrainingBatch{Frames: 2, VoiceFrames: 1, VoiceLatents: []float32{.1, .2}, NormalizedLatents: []float32{.3, .4, .5, .6}, TextTokens: []uint32{1, 1}}
	seed := make([]float32, 18)
	if _, err := m.BuildConditionerTrainingTape(batch, seed, 1); err == nil {
		t.Fatal("budget")
	}
	for _, which := range []string{"voice", "audio", "seed", "token", "bias"} {
		m := conditionerFixture()
		b := batch
		b.VoiceLatents = append([]float32(nil), batch.VoiceLatents...)
		b.NormalizedLatents = append([]float32(nil), batch.NormalizedLatents...)
		s := append([]float32(nil), seed...)
		switch which {
		case "voice":
			b.VoiceLatents[0] = float32(math.NaN())
		case "audio":
			b.NormalizedLatents[0] = float32(math.NaN())
		case "seed":
			s[0] = float32(math.NaN())
		case "token":
			b.TextTokens = []uint32{99, 1}
		case "bias":
			m.Input.Bias = make([]float32, 3)
		}
		if _, err := m.BuildConditionerTrainingTape(b, s, 1<<20); err == nil {
			t.Fatal(which)
		}
	}
	p, err := m.BuildConditionerTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	before := interpretFlowTape(t, p.Tape)
	batch.VoiceLatents[0] = 99
	batch.NormalizedLatents[0] = 99
	seed[0] = 99
	m.Input.Weight[0] = 99
	m.SpeakerProjection.Weight[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
}
