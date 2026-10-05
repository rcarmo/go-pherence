package pockettts

import (
	"encoding/json"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

func TestTrainingTransformerTapePyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("transformer oracle provenance")
	}
	model := transformerFromOracle(t, oracle)
	tape, err := model.BuildTransformerTrainingTape(oracle.Sequence, oracle.DOutput, oracle.Rows, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	result, err := tape.Decode(interpretFlowTape(t, tape.Tape))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "transformer tape PyTorch output", result.Output, oracle.Output, 4e-6)
	assertSliceClose(t, "transformer tape PyTorch input", result.DSequence, oracle.DSequence, 8e-6)
	if len(result.Parameters) != len(oracle.Gradients) {
		t.Fatal("gradient coverage", len(result.Parameters), len(oracle.Gradients))
	}
	for name, got := range result.Parameters {
		key := strings.TrimPrefix(name, "flow_lm.")
		if strings.HasPrefix(key, "out_norm.") {
			key = "final." + strings.TrimPrefix(key, "out_norm.")
		}
		want, ok := oracle.Gradients[key]
		if !ok {
			t.Fatal("oracle lacks", key)
		}
		assertSliceClose(t, "transformer tape PyTorch "+key, got, want, 1e-5)
	}
	t.Logf("transformer tape tensors=%d commands=%d gradients=%d", len(tape.Tape.Tensors), len(tape.Tape.Commands), len(result.Parameters))
}
func TestTrainingTransformerTapeTopology(t *testing.T) {
	for _, contextRows := range []int{0, 1, 2, 128} {
		for _, layerScale := range []bool{false, true} {
			for _, final := range []bool{false, true} {
				m := tinyFlowLMTraining().Transformer
				if !layerScale {
					m.Layers[0].LayerScale1 = nil
					m.Layers[0].LayerScale2 = nil
				}
				if !final {
					m.FinalWeight, m.FinalBias = nil, nil
				}
				m.Layers = append(m.Layers, m.Layers[0])
				m.Context = contextRows
				rows := 3
				x, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
				for i := range x {
					x[i] = float32(i%7-3) * .1
					seed[i] = float32(i%5-2) * .2
				}
				expected, grad, dx, err := m.ForwardBackward(x, seed, rows)
				if err != nil {
					t.Fatal(err)
				}
				tape, err := m.BuildTransformerTrainingTape(x, seed, rows, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				result, err := tape.Decode(interpretFlowTape(t, tape.Tape))
				if err != nil {
					t.Fatal(err)
				}
				assertSliceClose(t, "transformer tape topology output", result.Output, expected, 4e-6)
				assertSliceClose(t, "transformer tape topology input", result.DSequence, dx, 8e-6)
				named := map[string][]float32{}
				addFlowLMGradients(named, &FlowLMTrainingGradients{Transformer: grad})
				for name, got := range result.Parameters {
					assertSliceClose(t, "transformer tape topology "+name, got, named[name], 1e-5)
				}
			}
		}
	}
}

func TestTrainingTransformerShapeViews(t *testing.T) {
	g := &transformerTapeGraph{}
	source := g.tensor([]int{2, 3})
	if got := g.reshape(source, 2, 3); got != source || len(g.tape.Commands) != 0 {
		t.Fatal("identity shape view", got)
	}
	flat := g.reshape(source, 6)
	if again := g.reshape(source, 6); again != flat || len(g.tape.Commands) != 1 {
		t.Fatal("repeated shape view", again, flat)
	}
	other := g.tensor([]int{2, 3})
	if got := g.reshape(other, 6); got == flat {
		t.Fatal("distinct sources merged")
	}
	if got := g.reshape(source, 3, 2); got == source || got == flat {
		t.Fatal("different dimensions merged")
	}
}

func TestTrainingTransientZeroGraphParity(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	model := transformerFromOracle(t, oracle)
	for _, transient := range []bool{false, true} {
		g := &transformerTapeGraph{borrowParameters: true, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, transientZeros: transient}
		input := g.constant([]int{oracle.Rows, model.Width}, oracle.Sequence)
		seed := g.constant([]int{oracle.Rows, model.Width}, oracle.DOutput)
		out, dx, gradients, err := model.appendTransformerTrainingGraph(g, input, seed, oracle.Rows)
		if err != nil {
			t.Fatal(err)
		}
		names := make([]string, 0, len(g.params))
		for name := range g.params {
			names = append(names, name)
		}
		sort.Strings(names)
		bindings := map[string]train.ResidentTapeBinding{}
		g.tape.Outputs = []int{out, dx}
		for _, name := range names {
			id := g.params[name]
			bindings[name] = train.ResidentTapeBinding{Parameter: id, Gradient: gradients[id]}
			g.tape.Outputs = append(g.tape.Outputs, gradients[id])
		}
		built, err := materializeSequenceTape(g, bindings, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		values := interpretFlowTape(t, built.Tape)
		assertSliceClose(t, "transient output", values[0], oracle.Output, 4e-6)
		assertSliceClose(t, "transient input gradient", values[1], oracle.DSequence, 8e-6)
		for i, name := range names {
			key := strings.TrimPrefix(name, "flow_lm.")
			if strings.HasPrefix(key, "out_norm.") {
				key = "final." + strings.TrimPrefix(key, "out_norm.")
			}
			assertSliceClose(t, "transient parameter "+name, values[i+2], oracle.Gradients[key], 1e-5)
		}
		if _, err = train.PlanZeroTapeStorage(built.Tape, 64<<20, map[int]bool{}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTrainingTransientZeroFillContract(t *testing.T) {
	g := &transformerTapeGraph{borrowParameters: true, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, transientZeros: true}
	a, b := g.zero(2, 3), g.zero(2, 3)
	if a == b {
		t.Fatal("computed fills interned across lifetimes")
	}
	literal := g.uniqueZero(6)
	g.tape.Outputs = []int{a, b, literal}
	built, err := materializeSequenceTape(g, nil, 1024)
	if err != nil {
		t.Fatal(err)
	}
	values := interpretFlowTape(t, built.Tape)
	for _, v := range values {
		for _, x := range v {
			if math.Float32bits(x) != 0 {
				t.Fatal("positive zero contract", x)
			}
		}
	}
	if built.Tape.Tensors[literal].Values == nil {
		t.Fatal("literal placeholder became computed")
	}
}

func TestTrainingTransformerLayerStageOriginalIdentity(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	model := transformerFromOracle(t, oracle)
	model.Layers = append(model.Layers, model.Layers[0])
	for _, forward := range []bool{false, true} {
		stage, err := buildTransformerLayerStage(model, 1, oracle.Sequence, oracle.DOutput, oracle.Rows, forward, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range stage.ParameterNames {
			if !strings.HasPrefix(name, "flow_lm.transformer.layers.1.") {
				t.Fatal("stage parameter identity", name)
			}
		}
		subset := *model
		subset.Layers = model.Layers[1:2]
		subset.FinalWeight = nil
		subset.FinalBias = nil
		want, grads, dx, err := subset.ForwardBackward(oracle.Sequence, oracle.DOutput, oracle.Rows)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, stage.Tape)
		assertSliceClose(t, "layer stage output", got[0], want, 4e-6)
		if !forward {
			assertSliceClose(t, "layer stage input VJP", got[1], dx, 8e-6)
			expected := map[string][]float32{}
			addFlowLMGradients(expected, &FlowLMTrainingGradients{Transformer: grads})
			for i, name := range stage.ParameterNames {
				local := strings.Replace(name, "layers.1.", "layers.0.", 1)
				assertSliceClose(t, "layer stage gradient "+name, got[i+2], expected[local], 1e-5)
			}
		}
	}
	// Omitted original layers still reject invalid parameters before a stage.
	model.Layers[0].Norm1Weight = append([]float32(nil), model.Layers[0].Norm1Weight...)
	model.Layers[0].Norm1Weight[0] = float32(math.NaN())
	if _, err := buildTransformerLayerStage(model, 1, oracle.Sequence, oracle.DOutput, oracle.Rows, false, 64<<20); err == nil {
		t.Fatal("invalid omitted model parameter admitted")
	}
}

func TestTrainingTransformerPartitionedCPUReference(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	m := transformerFromOracle(t, oracle)
	m.Layers = append(m.Layers, m.Layers[0])
	want, grads, dx, err := m.ForwardBackward(oracle.Sequence, oracle.DOutput, oracle.Rows)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string][]float32{}
	addFlowLMGradients(expected, &FlowLMTrainingGradients{Transformer: grads})
	boundaries := [][]float32{append([]float32(nil), oracle.Sequence...)}
	for i := range m.Layers {
		stage, err := buildTransformerLayerStage(m, i, boundaries[i], make([]float32, len(oracle.Sequence)), oracle.Rows, true, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		values := interpretFlowTape(t, stage.Tape)
		boundaries = append(boundaries, values[0])
	}
	final, err := buildTransformerFinalNormStage(m, boundaries[len(boundaries)-1], make([]float32, len(oracle.Sequence)), oracle.Rows, true, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	got := interpretFlowTape(t, final.Tape)
	assertSliceClose(t, "partitioned output", got[0], want, 4e-6)
	reverse, err := buildTransformerFinalNormStage(m, boundaries[len(boundaries)-1], oracle.DOutput, oracle.Rows, false, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	rv := interpretFlowTape(t, reverse.Tape)
	seed := rv[1]
	for i, name := range reverse.ParameterNames {
		assertSliceClose(t, "partitioned final "+name, rv[i+2], expected[name], 1e-5)
	}
	for layer := len(m.Layers) - 1; layer >= 0; layer-- {
		stage, err := buildTransformerLayerStage(m, layer, boundaries[layer], seed, oracle.Rows, false, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		values := interpretFlowTape(t, stage.Tape)
		seed = values[1]
		for i, name := range stage.ParameterNames {
			assertSliceClose(t, "partitioned parameter "+name, values[i+2], expected[name], 1e-5)
		}
	}
	assertSliceClose(t, "partitioned input VJP", seed, dx, 8e-6)
}
