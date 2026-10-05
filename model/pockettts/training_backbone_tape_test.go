package pockettts

import (
	"encoding/json"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"os"
	"testing"
)

func backboneOracle(t *testing.T) (flowLMTrainingOracle, *FlowLMTrainingCPU) {
	t.Helper()
	data, err := os.ReadFile("testdata/flowlm_training_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var o flowLMTrainingOracle
	if err = json.Unmarshal(data, &o); err != nil {
		t.Fatal(err)
	}
	if o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle pin")
	}
	return o, flowLMTrainingFromOracle(t, o)
}
func TestTrainingBackboneTapePyTorchParity(t *testing.T) {
	o, m := backboneOracle(t)
	batch := FlowLMTrainingBatch{o.Frames, o.VoiceFrames, o.NormalizedLatents, o.VoiceLatents, o.TextTokens}
	// The independent PyTorch FlowLM oracle supplies EOS contributions to the
	// backbone seed. EOS parameter gradients remain outside this fused tape.
	rows := 1 + o.VoiceFrames + len(o.TextTokens) + o.Frames
	prefix := rows - o.Frames
	seed := make([]float32, rows*m.Hidden)
	for row := 0; row < o.Frames; row++ {
		for k := 0; k < m.Hidden; k++ {
			seed[(prefix+row)*m.Hidden+k] = o.DZ[row*m.Hidden+k] + o.DEOS[row]*m.EOS.Weight[k]
		}
	}
	p, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	assertSliceClose(t, "sequence", v[0], o.Sequence, 2e-6)
	assertSliceClose(t, "Z", v[1][prefix*m.Hidden:], o.Z, 5e-6)
	assertSliceClose(t, "voice", v[2], o.DVoiceLatents, 1.5e-5)
	assertSliceClose(t, "normalized", v[3], o.DNormalizedLatents, 1.5e-5)
	values := map[int][]float32{}
	for i, id := range p.Tape.Outputs {
		values[id] = v[i]
	}
	if len(p.ResidentBindings) != len(o.Gradients)-2 {
		t.Fatal("parameter coverage")
	}
	for name, binding := range p.ResidentBindings {
		want, ok := o.Gradients[name[len("flow_lm."):]]
		if !ok {
			t.Fatal(name)
		}
		assertSliceClose(t, name, values[binding.Gradient], want, 2e-5)
	}
}
func TestTrainingBackboneTapeComposition(t *testing.T) {
	for _, frames := range []int{1, 2, 7} {
		for _, voice := range []int{0, 1, 3} {
			for _, tokens := range [][]uint32{nil, {1, 1, 2}} {
				_, m := backboneOracle(t)
				batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: voice, NormalizedLatents: make([]float32, frames*m.LatentDim), VoiceLatents: make([]float32, voice*m.LatentDim), TextTokens: tokens}
				for i := range batch.NormalizedLatents {
					batch.NormalizedLatents[i] = float32(i%11-5) * .08
				}
				for i := range batch.VoiceLatents {
					batch.VoiceLatents[i] = float32(i%7-3) * .1
				}
				rows := 1 + voice + len(tokens) + frames
				seed := make([]float32, rows*m.Hidden)
				for i := range seed {
					seed[i] = float32(i%11-5) * .03
				}
				cp, err := m.BuildConditionerTrainingTape(batch, make([]float32, len(seed)), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				cv := interpretFlowTape(t, cp.Tape)
				// Independent host transformer implementation, not the extracted graph.
				tr, tg, dx, err := m.Transformer.ForwardBackward(cv[0], seed, rows)
				if err != nil {
					t.Fatal(err)
				}
				cb, err := m.BuildConditionerTrainingTape(batch, dx, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				bv := interpretFlowTape(t, cb.Tape)
				bp, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, bp.Tape)
				assertSliceClose(t, "sequence", got[0], cv[0], 2e-6)
				assertSliceClose(t, "transformer", got[1], tr, 5e-6)
				n := 1
				if voice > 0 {
					n = 2
					assertSliceClose(t, "rawvoice", got[2], bv[1], 1.5e-5)
				}
				assertSliceClose(t, "normalized", got[n+1], bv[n], 1.5e-5)
				values := map[int][]float32{}
				for i, id := range bp.Tape.Outputs {
					values[id] = got[i]
				}
				want := map[string][]float32{}
				addFlowLMGradients(want, &FlowLMTrainingGradients{Transformer: tg})
				for i, id := range cb.Tape.Outputs {
					for name, binding := range cb.ResidentBindings {
						if binding.Gradient == id {
							want[name] = bv[i]
						}
					}
				}
				for name, binding := range bp.ResidentBindings {
					assertSliceClose(t, name, values[binding.Gradient], want[name], 2e-5)
				}
			}
		}
	}
}
func TestTrainingBackboneTapeTopology(t *testing.T) {
	for _, context := range []int{0, 1, 2, 128} {
		for _, scale := range []bool{false, true} {
			for _, final := range []bool{false, true} {
				_, m := backboneOracle(t)
				m.Transformer.Context = context
				if !scale {
					for i := range m.Transformer.Layers {
						m.Transformer.Layers[i].LayerScale1 = nil
						m.Transformer.Layers[i].LayerScale2 = nil
					}
				}
				if !final {
					m.Transformer.FinalWeight, m.Transformer.FinalBias = nil, nil
				}
				batch := FlowLMTrainingBatch{Frames: 2, NormalizedLatents: make([]float32, 2*m.LatentDim), TextTokens: []uint32{1}}
				seed := make([]float32, 4*m.Hidden)
				for i := range seed {
					seed[i] = float32(i%7-3) * .03
				}
				cp, err := m.BuildConditionerTrainingTape(batch, make([]float32, len(seed)), 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				cv := interpretFlowTape(t, cp.Tape)
				want, _, dx, err := m.Transformer.ForwardBackward(cv[0], seed, 4)
				if err != nil {
					t.Fatal(err)
				}
				cb, err := m.BuildConditionerTrainingTape(batch, dx, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				bv := interpretFlowTape(t, cb.Tape)
				p, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				v := interpretFlowTape(t, p.Tape)
				assertSliceClose(t, "topology transformer", v[1], want, 5e-6)
				assertSliceClose(t, "topology input", v[2], bv[1], 1.5e-5)
			}
		}
	}
}
func TestTrainingBackboneTapeAdmissionAndOwnership(t *testing.T) {
	_, m := backboneOracle(t)
	batch := FlowLMTrainingBatch{Frames: 2, VoiceFrames: 1, NormalizedLatents: make([]float32, 2*m.LatentDim), VoiceLatents: make([]float32, m.LatentDim), TextTokens: []uint32{1, 1}}
	seed := make([]float32, 6*m.Hidden)
	p, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	for _, budget := range []int64{0, -1, bytes - 1} {
		if _, err = m.BuildBackboneTrainingTape(batch, seed, budget); err == nil {
			t.Fatal("budget", budget)
		}
	}
	if _, err = m.BuildBackboneTrainingTape(batch, seed, bytes); err != nil {
		t.Fatal("exact budget", err)
	}
	before := interpretFlowTape(t, p.Tape)
	batch.VoiceLatents[0] = 99
	batch.NormalizedLatents[0] = 99
	seed[0] = 99
	m.Input.Weight[0] = 99
	m.Transformer.Layers[0].FC1.Weight[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	for _, kind := range []string{"voice", "audio", "seed", "token", "transformer", "width", "layers", "context", "period", "parameter", "embedding"} {
		t.Run(kind, func(t *testing.T) {
			_, m := backboneOracle(t)
			b := FlowLMTrainingBatch{Frames: 1, NormalizedLatents: make([]float32, m.LatentDim)}
			s := make([]float32, 2*m.Hidden)
			switch kind {
			case "voice":
				b.VoiceFrames = 1
				b.VoiceLatents = []float32{float32(math.NaN())}
			case "audio":
				b.NormalizedLatents[0] = float32(math.NaN())
			case "seed":
				s[0] = float32(math.Inf(1))
			case "token":
				b.TextTokens = []uint32{uint32(m.Vocabulary)}
				s = make([]float32, 3*m.Hidden)
			case "transformer":
				m.Transformer = nil
			case "width":
				m.Transformer.Width++
			case "layers":
				for len(m.Transformer.Layers) <= 16 {
					m.Transformer.Layers = append(m.Transformer.Layers, m.Transformer.Layers[0])
				}
			case "context":
				m.Transformer.Context = -1
			case "period":
				m.Transformer.MaxPeriod = math.Inf(1)
			case "parameter":
				m.Transformer.Layers[0].FC1.Weight[0] = float32(math.NaN())
			case "embedding":
				m.Embedding[0] = float32(math.NaN())
			}
			if _, err := m.BuildBackboneTrainingTape(b, s, 64<<20); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	// Private read-only named parameters, but no caller-input borrowing.
	_, m = backboneOracle(t)
	batch.VoiceLatents = make([]float32, m.LatentDim)
	batch.NormalizedLatents = make([]float32, 2*m.LatentDim)
	clear(seed)
	private, err := m.buildBackboneTrainingTape(batch, seed, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	parameters := map[string][]float32{"flow_lm.conditioner.embed.weight": m.Embedding, "flow_lm.input_linear.weight": m.Input.Weight}
	for name, x := range parameters {
		if !f32SlicesOverlap(private.Tape.Tensors[private.ResidentBindings[name].Parameter].Values, x) {
			t.Fatal("not borrowed", name)
		}
	}
	for _, s := range private.Tape.Tensors {
		for _, x := range [][]float32{batch.VoiceLatents, batch.NormalizedLatents, seed} {
			if f32SlicesOverlap(s.Values, x) {
				t.Fatal("borrowed nonparam")
			}
		}
	}
	// Complete source admission still rejects dead constants before pruning.
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20); err == nil {
		t.Fatal("dead invalid source")
	}
}

func TestTrainingBackboneResultGuards(t *testing.T) {
	_, m := backboneOracle(t)
	batch := FlowLMTrainingBatch{Frames: 2, VoiceFrames: 1, NormalizedLatents: make([]float32, 2*m.LatentDim), VoiceLatents: make([]float32, m.LatentDim)}
	seed := make([]float32, 4*m.Hidden)
	fresh := func() BackboneTrainingResult {
		return BackboneTrainingResult{make([]float32, len(seed)), make([]float32, len(seed)), make([]float32, m.LatentDim), make([]float32, 2*m.LatentDim)}
	}
	if err := validateBackboneResult(fresh(), m, batch, seed); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"shape", "nan", "source", "overlap", "tail"} {
		t.Run(kind, func(t *testing.T) {
			r := fresh()
			switch kind {
			case "shape":
				r.Sequence = nil
			case "nan":
				r.Output[0] = float32(math.NaN())
			case "source":
				r.Sequence = seed
			case "overlap":
				r.Output = r.Sequence
			case "tail":
				r.DNormalized[len(r.DNormalized)-1] = 1
			}
			if err := validateBackboneResult(r, m, batch, seed); err == nil {
				t.Fatal("bad result")
			}
		})
	}
}

func TestTrainingBackboneTapeMaximumLayers(t *testing.T) {
	_, m := backboneOracle(t)
	for len(m.Transformer.Layers) < 16 {
		m.Transformer.Layers = append(m.Transformer.Layers, m.Transformer.Layers[0])
	}
	batch := FlowLMTrainingBatch{Frames: 1, NormalizedLatents: make([]float32, m.LatentDim)}
	seed := make([]float32, 2*m.Hidden)
	p, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Tape.Tensors) > 4096 || len(p.Tape.Commands) > 4096 {
		t.Fatal("graph ceiling")
	}
	t.Logf("16-layer tiny backbone tensors %d commands %d", len(p.Tape.Tensors), len(p.Tape.Commands))
}
