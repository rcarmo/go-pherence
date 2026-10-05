package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingEOSBackboneTapeComposition(t *testing.T) {
	for _, frames := range []int{1, 2, 7} {
		for _, voice := range []int{0, 1} {
			for _, weight := range []float32{0, .1, 1, 2} {
				_, m := backboneOracle(t)
				batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: voice, NormalizedLatents: make([]float32, frames*m.LatentDim), VoiceLatents: make([]float32, voice*m.LatentDim), TextTokens: []uint32{1, 1}}
				for i := range batch.NormalizedLatents {
					batch.NormalizedLatents[i] = float32(i%11-5) * .08
				}
				for i := range batch.VoiceLatents {
					batch.VoiceLatents[i] = float32(i%7-3) * .1
				}
				rows := 1 + voice + 2 + frames
				prefix := rows - frames
				seed := make([]float32, rows*m.Hidden)
				for i := range seed {
					seed[i] = float32(i%11-5) * .03
				}
				forward, err := m.BuildBackboneTrainingTape(batch, make([]float32, len(seed)), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				fv := interpretFlowTape(t, forward.Tape)
				hidden := fv[1][prefix*m.Hidden:]
				mask := make([]bool, frames)
				for i := range mask {
					mask[i] = i < max(1, frames-2)
				}
				ep, err := m.BuildEOSProjectionTrainingTape(hidden, mask, weight, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				ev := interpretFlowTape(t, ep.Tape)
				combined := append([]float32(nil), seed...)
				for i, v := range ev[1] {
					combined[prefix*m.Hidden+i] += v
				}
				reference, err := m.BuildBackboneTrainingTape(batch, combined, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				rv := interpretFlowTape(t, reference.Tape)
				p, err := m.BuildEOSBackboneTrainingTape(batch, seed, hidden, mask, weight, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, p.Tape)
				count := 3
				if voice > 0 {
					count = 4
				}
				for i := 0; i < count; i++ {
					assertSliceClose(t, "EOS backbone boundary", got[i], rv[i], 1.5e-5)
				}
				values := map[int][]float32{}
				for i, id := range p.Tape.Outputs {
					values[id] = got[i]
				}
				want := map[string][]float32{}
				for i, id := range reference.Tape.Outputs {
					for name, binding := range reference.ResidentBindings {
						if binding.Gradient == id {
							want[name] = rv[i]
						}
					}
				}
				for i, id := range ep.Tape.Outputs {
					for name, binding := range ep.ResidentBindings {
						if binding.Gradient == id {
							want[name] = ev[i]
						}
					}
				}
				if len(p.ResidentBindings) != len(reference.ResidentBindings)+2 {
					t.Fatal("EOS gradient inventory")
				}
				for name, binding := range p.ResidentBindings {
					assertSliceClose(t, name, values[binding.Gradient], want[name], 2e-5)
				}
				if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
func TestTrainingEOSBackboneTapePyTorchParity(t *testing.T) {
	o, m := backboneOracle(t)
	batch := FlowLMTrainingBatch{o.Frames, o.VoiceFrames, o.NormalizedLatents, o.VoiceLatents, o.TextTokens}
	rows := 1 + o.VoiceFrames + len(o.TextTokens) + o.Frames
	prefix := rows - o.Frames
	seed := make([]float32, rows*m.Hidden)
	copy(seed[prefix*m.Hidden:], o.DZ)
	mask := make([]bool, o.Frames)
	for i := range mask {
		mask[i] = true
	}
	// PyTorch oracle dEOS is arbitrary, so compare against independently seeded
	// full CPU FlowLM using BCE labels for this new composed loss policy.
	logits := make([]float32, o.Frames)
	for row := range logits {
		logits[row] = linearForwardTraining(m.EOS, o.Z[row*m.Hidden:(row+1)*m.Hidden])[0]
	}
	_, de, err := EOSLossAndGradient(logits, mask)
	if err != nil {
		t.Fatal(err)
	}
	for i := range de {
		de[i] *= .1
	}
	output, g, inputs, err := m.ForwardBackward(batch, o.DZ, de)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildEOSBackboneTrainingTape(batch, seed, o.Z, mask, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	assertSliceClose(t, "sequence", v[0], o.Sequence, 2e-6)
	assertSliceClose(t, "Z", v[1][prefix*m.Hidden:], o.Z, 5e-6)
	assertSliceClose(t, "voice", v[2], inputs.VoiceLatents, 1.5e-5)
	assertSliceClose(t, "normalized", v[3], inputs.NormalizedLatents, 1.5e-5)
	assertSliceClose(t, "CPU Z", v[1][prefix*m.Hidden:], output.Z, 5e-6)
	wanted := flowLMTrainingGradientMap(g)
	values := map[int][]float32{}
	for i, id := range p.Tape.Outputs {
		values[id] = v[i]
	}
	for name, binding := range p.ResidentBindings {
		assertSliceClose(t, name, values[binding.Gradient], wanted[name[len("flow_lm."):]], 2e-5)
	}
}
func TestTrainingEOSBackboneTapeAdmissionOwnership(t *testing.T) {
	_, m := backboneOracle(t)
	batch := FlowLMTrainingBatch{Frames: 1, NormalizedLatents: make([]float32, m.LatentDim)}
	seed := make([]float32, 2*m.Hidden)
	hidden := make([]float32, m.Hidden)
	p, err := m.BuildEOSBackboneTrainingTape(batch, seed, hidden, []bool{true}, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err = m.BuildEOSBackboneTrainingTape(batch, seed, hidden, []bool{true}, .1, bytes-1); err == nil {
		t.Fatal("full bytes")
	}
	if _, err = m.BuildEOSBackboneTrainingTape(batch, seed, hidden, []bool{true}, .1, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"maskshape", "noactive", "weight", "hidden", "overflow", "bias", "seed", "budget"} {
		t.Run(kind, func(t *testing.T) {
			_, m := backboneOracle(t)
			h := make([]float32, m.Hidden)
			s := make([]float32, 2*m.Hidden)
			mask := []bool{true}
			w := float32(.1)
			budget := int64(64 << 20)
			switch kind {
			case "maskshape":
				mask = nil
			case "noactive":
				mask[0] = false
			case "weight":
				w = -1
			case "hidden":
				h[0] = float32(math.NaN())
			case "overflow":
				h[0] = math.MaxFloat32
				m.EOS.Weight[0] = 4
			case "bias":
				m.EOS.Bias = nil
			case "seed":
				s[0] = float32(math.NaN())
			case "budget":
				budget = 0
			}
			if _, err := m.BuildEOSBackboneTrainingTape(batch, s, h, mask, w, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	m.EOS.Weight[0] = 99
	m.EOS.Bias[0] = 99
	batch.NormalizedLatents[0] = 99
	seed[0] = 99
	hidden[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
}
