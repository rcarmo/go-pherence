package pockettts

import (
	"encoding/json"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"os"
	"reflect"
	"testing"
)

type parameterBorrowPlan struct {
	tape     train.Tape
	bindings map[string]train.ResidentTapeBinding
}

func TestTrainingResidentParameterBorrowBuilders(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	params, err := fullParameterMap(lm, flow, w)
	if err != nil {
		t.Fatal(err)
	}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, TextTokens: oracle.TextTokens, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents}
	cases := []struct {
		name  string
		build func(bool, int64) (parameterBorrowPlan, error)
	}{
		{"flow-host-time", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := flow.buildFlowDualTrainingTape(make([]float32, flow.Condition.In), []float32{.2, .7}, make([]float32, flow.Input.In), 1, make([]float32, flow.Final.Linear.Out), make([]float32, flow.Final.Linear.Out), limit, false, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"flow-device-time", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := flow.buildFlowDualTrainingTape(make([]float32, flow.Condition.In), []float32{.2, .7}, make([]float32, flow.Input.In), 1, make([]float32, flow.Final.Linear.Out), make([]float32, flow.Final.Linear.Out), limit, true, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"transformer", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := lm.Transformer.buildTransformerTrainingTape(make([]float32, 2*lm.Hidden), make([]float32, 2*lm.Hidden), 2, limit, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"weight", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := w.buildWeightTrainingTape([]float32{.2, .3, .4, .5}, []float32{.3, -.4}, 2, limit, nil, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"sampleweight", func(b bool, limit int64) (parameterBorrowPlan, error) {
			s := WeightSampleInputs{[]float32{.2, .3}, []float32{.1, .2}, []float32{.6, .8}, []float32{.3, -.2}, []float32{-.1, .4}}
			p, e := w.buildWeightTrainingTape(nil, nil, 4, limit, &s, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"projection", func(b bool, limit int64) (parameterBorrowPlan, error) {
			l := LinearF32{In: lm.LatentDim, Out: lm.Hidden, Weight: lm.Input.Weight}
			p, e := buildProjectionTrainingTape(l, "flow_lm.input_linear.weight", "", make([]float32, 2*l.In), make([]float32, 2*l.Out), 2, limit, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"audio", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := lm.buildAudioAssemblyTrainingTape(batch, make([]float32, batch.Frames*lm.LatentDim), limit, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
		{"sequence", func(b bool, limit int64) (parameterBorrowPlan, error) {
			p, e := lm.buildSequenceTrainingTape(batch, make([]float32, batch.VoiceFrames*lm.Hidden), make([]float32, batch.Frames*lm.Hidden), make([]float32, (1+batch.VoiceFrames+len(batch.TextTokens)+batch.Frames)*lm.Hidden), limit, b)
			return parameterBorrowPlan{p.Tape, p.ResidentBindings}, e
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			owned, err := c.build(false, 1<<24)
			if err != nil {
				t.Fatal(err)
			}
			borrowed, err := c.build(true, 1<<24)
			if err != nil {
				t.Fatal(err)
			}
			if err = train.ValidateResidentTapeBindings(borrowed.tape, borrowed.bindings, 1<<24); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(owned.tape.Commands, borrowed.tape.Commands) || !reflect.DeepEqual(owned.tape.Outputs, borrowed.tape.Outputs) || !reflect.DeepEqual(owned.bindings, borrowed.bindings) || len(owned.tape.Tensors) != len(borrowed.tape.Tensors) {
				t.Fatal("metadata/topology changed")
			}
			bound := map[int]bool{}
			for n, x := range borrowed.bindings {
				src := params[n]
				if len(src) == 0 {
					t.Fatal("source inventory", n)
				}
				bound[x.Parameter] = true
				if !f32SlicesOverlap(borrowed.tape.Tensors[x.Parameter].Values, src) || f32SlicesOverlap(owned.tape.Tensors[x.Parameter].Values, src) {
					t.Fatal("ownership contract", n)
				}
			}
			for id, x := range owned.tape.Tensors {
				y := borrowed.tape.Tensors[id]
				if !reflect.DeepEqual(x.Shape, y.Shape) {
					t.Fatal("tensor shape")
				}
				assertSliceClose(t, "constant", x.Values, y.Values, 0)
				if !bound[id] && len(x.Values) > 0 && f32SlicesOverlap(x.Values, y.Values) {
					t.Fatal("nonparameter constant borrowed", id)
				}
			}
			a, b := interpretFlowTape(t, owned.tape), interpretFlowTape(t, borrowed.tape)
			for i, x := range a {
				assertSliceClose(t, "owned/borrowed arithmetic", b[i], x, 2e-7)
			}
			if _, err = c.build(true, 1); err == nil {
				t.Fatal("full source byte ceiling bypassed")
			}
			// Validate borrowed current host values, even though device bindings replace
			// them later; malformed unused host parameters must never bypass admission.
			for n := range borrowed.bindings {
				x := params[n]
				old := x[0]
				x[0] = float32(math.NaN())
				if _, err = c.build(true, 1<<24); err == nil {
					t.Fatal("nonfinite parameter admission", n)
				}
				x[0] = old
				break
			}
		})
	}
}
func TestTrainingResidentParameterBorrowConstantOwnership(t *testing.T) {
	l := LinearF32{In: 2, Out: 2, Weight: []float32{1, 2, 3, 4}, Bias: []float32{.1, .2}}
	input, seed := []float32{.2, .3}, []float32{.4, .5}
	p, err := buildProjectionTrainingTape(l, "w", "b", input, seed, 1, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	before := interpretFlowTape(t, p.Tape)
	input[0] = 99
	seed[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "nonparam immutable", after[i], x, 0)
	}
	// Public builder must keep its parameters independent even after this private
	// path adds read-only borrowing as an internal construction optimization.
	public, err := BuildProjectionTrainingTape(l, "w", "b", input, seed, 1, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	old := append([]float32(nil), public.Tape.Tensors[public.ResidentBindings["w"].Parameter].Values...)
	l.Weight[0] = 9
	assertSliceClose(t, "public owned parameter", public.Tape.Tensors[public.ResidentBindings["w"].Parameter].Values, old, 0)
	if p.Tape.Tensors[p.ResidentBindings["w"].Parameter].Values[0] != 9 {
		t.Fatal("private source provenance")
	}
}
func BenchmarkTrainingResidentParameterBorrow(b *testing.B) {
	raw, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		b.Fatal(err)
	}
	var oracle trainingStepOracle
	if err = json.Unmarshal(raw, &oracle); err != nil {
		b.Fatal(err)
	}
	for _, mode := range []string{"owned", "borrowed"} {
		b.Run(mode, func(b *testing.B) {
			_, _, w := trainingStepModelsFromOracle(b, oracle)
			w.Layers = []LinearF32{{In: 2, Out: 256, Weight: make([]float32, 512), Bias: make([]float32, 256)}, {In: 256, Out: 256, Weight: make([]float32, 65536), Bias: make([]float32, 256)}, {In: 256, Out: 1, Weight: make([]float32, 256), Bias: make([]float32, 1)}}
			x, seed := []float32{.2, .3, .4, .5}, []float32{.3, -.4}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err = w.buildWeightTrainingTape(x, seed, 2, 1<<24, nil, mode == "borrowed"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
