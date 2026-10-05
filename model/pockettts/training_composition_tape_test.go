package pockettts

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func distillCompositionSources(t *testing.T) (EndpointTrainingTape, PrimaryTrainingTape) {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := distillFixture(t, m, .3, .8)
	ep, err := m.BuildEndpointTrainingTape(in.Condition, in.T, in.EndpointInput, in.Prediction, in.Desired, in.Velocity, in.Tangent, in.LogVariance, in.T-in.S, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	n := len(in.Target)
	pp, err := m.BuildPrimaryTrainingTape(PrimaryTrainingInputs{Condition: in.Condition, Input: in.PrimaryInput, GV: make([]float32, n), GT: make([]float32, n), EndpointInput: make([]float32, n), Target: in.Target, Noise: in.Noise, S: in.S, T: in.T}, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return ep, pp
}
func compositionConstant(tape *train.Tape, n int) int {
	id := len(tape.Tensors)
	tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{n}, Values: make([]float32, n)})
	return id
}
func compositionExtraBinding(tape *train.Tape, bindings map[string]train.ResidentTapeBinding, name string, n int) {
	parameter := compositionConstant(tape, n)
	gradient := compositionConstant(tape, n)
	bindings[name] = train.ResidentTapeBinding{Parameter: parameter, Gradient: gradient}
	tape.Outputs = append(tape.Outputs, gradient)
}

func TestTrainingDistillCompositionGuards(t *testing.T) {
	cases := map[string]string{"primary-source": "nonfinite input", "endpoint-source": "nonfinite input", "outputs": "source outputs", "inventory": "parameter inventory", "missing": "missing endpoint parameter", "extent": "parameter extent", "seed-id": "seed edge", "seed-extent": "seed extent", "budget": "byte ceiling"}
	for kind, want := range cases {
		t.Run(kind, func(t *testing.T) {
			ep, pp := distillCompositionSources(t)
			budget := int64(64 << 20)
			switch kind {
			case "primary-source":
				pp.Tape.Tensors = append(pp.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "endpoint-source":
				ep.Tape.Tensors = append(ep.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "outputs": // Admit a complete minimal binding inventory but reject missing boundary positions.
				ep = EndpointTrainingTape{Tape: train.Tape{}, ResidentBindings: map[string]train.ResidentTapeBinding{}}
				compositionExtraBinding(&ep.Tape, ep.ResidentBindings, "tiny.parameter", 1)
			case "inventory":
				compositionExtraBinding(&pp.Tape, pp.ResidentBindings, "additional.parameter", 1)
			case "missing":
				for name, b := range ep.ResidentBindings {
					delete(ep.ResidentBindings, name)
					ep.ResidentBindings["renamed.parameter"] = b
					break
				}
			case "extent": // Both source binding sets remain independently valid.
				compositionExtraBinding(&ep.Tape, ep.ResidentBindings, "additional.parameter", 1)
				compositionExtraBinding(&pp.Tape, pp.ResidentBindings, "additional.parameter", 2)
			case "seed-id":
				pp.seedIDs[0] = -1
			case "seed-extent":
				pp.seedIDs[0] = compositionConstant(&pp.Tape, 1)
			case "budget":
				full, err := composeDistillTrainingTapes(ep, pp, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				budget = tapeLogicalBytes(full.Tape) - 1
			}
			if _, err := composeDistillTrainingTapes(ep, pp, budget); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatal(kind, err)
			}
		})
	}
}
func tapeLogicalBytes(tape train.Tape) int64 {
	var bytes int64
	for _, s := range tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	return bytes
}

func forwardCompositionSources(t *testing.T) (train.Tape, FlowDualTrainingTape, FlowDualTrainingTape) {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	zero := make([]float32, len(in.Input))
	interpolation, err := BuildBranchTrainingTape("interpolate", [][]float32{in.Target, in.Noise}, .3, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{.3, .8}, zero, 1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{.8, .8}, zero, -1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return interpolation, p, e
}
func TestTrainingForwardCompositionGuards(t *testing.T) {
	cases := map[string]string{"delta": "forward delta", "interpolation-source": "nonfinite input", "primary-source": "nonfinite input", "endpoint-source": "nonfinite input", "outputs": "source outputs", "names-count": "parameter names", "names-missing": "parameter names", "names-duplicate": "parameter names", "primary-edge": "input edge", "endpoint-edge": "input edge", "endpoint-inventory": "parameter inventory", "budget": "byte ceiling"}
	for kind, want := range cases {
		t.Run(kind, func(t *testing.T) {
			interpolation, p, e := forwardCompositionSources(t)
			delta := float32(.5)
			budget := int64(64 << 20)
			switch kind {
			case "delta":
				delta = float32(math.NaN())
			case "interpolation-source":
				interpolation.Tensors = append(interpolation.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "primary-source":
				p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "endpoint-source":
				e.Tape.Tensors = append(e.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "outputs":
				p = FlowDualTrainingTape{Tape: train.Tape{}, ResidentBindings: map[string]train.ResidentTapeBinding{}}
				compositionExtraBinding(&p.Tape, p.ResidentBindings, "tiny.parameter", 1)
			case "names-count":
				p.ParameterNames = p.ParameterNames[:len(p.ParameterNames)-1]
			case "names-missing":
				p.ParameterNames[0] = "missing.parameter"
			case "names-duplicate":
				p.ParameterNames[0] = p.ParameterNames[1]
			case "primary-edge":
				p.inputValue = compositionConstant(&p.Tape, 1)
			case "endpoint-edge":
				e.inputValue = compositionConstant(&e.Tape, 1)
			case "endpoint-inventory":
				for name, b := range e.ResidentBindings {
					delete(e.ResidentBindings, name)
					e.ResidentBindings["renamed.parameter"] = b
					break
				}
			case "budget":
				full, err := composeDistillForwardTapes(interpolation, p, e, delta, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				budget = tapeLogicalBytes(full.Tape) - 1
			}
			if _, err := composeDistillForwardTapes(interpolation, p, e, delta, budget); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatal(kind, err)
			}
		})
	}
}

func TestTrainingCompositionSourcePreservation(t *testing.T) {
	ep, pp := distillCompositionSources(t)
	interpolation, p, e := forwardCompositionSources(t)
	// Retain deep copies of both source headers and backing arrays across success
	// and late combined-budget rejection. Composition never writes its sources.
	snapshot := func(x any) string {
		v, err := json.Marshal(x)
		if err != nil {
			t.Fatal(err)
		}
		return string(v)
	}
	seeds, inputIDs := pp.seedIDs, [2]int{p.inputValue, e.inputValue}
	before := []string{snapshot(ep), snapshot(pp), snapshot(interpolation), snapshot(p), snapshot(e)}
	distill, err := composeDistillTrainingTapes(ep, pp, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := composeDistillTrainingTapes(ep, pp, tapeLogicalBytes(distill.Tape)-1); err == nil {
		t.Fatal("budget")
	}
	if _, err := composeDistillForwardTapes(interpolation, p, e, .5, tapeLogicalBytes(forward.Tape)-1); err == nil {
		t.Fatal("budget")
	}
	after := []string{snapshot(ep), snapshot(pp), snapshot(interpolation), snapshot(p), snapshot(e)}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("source mutated")
	}
	// Primary seed IDs are private metadata; verify them explicitly too.
	if pp.seedIDs != seeds || [2]int{p.inputValue, e.inputValue} != inputIDs {
		t.Fatal("edge metadata")
	}
}

func TestTrainingForwardCompositionIDs(t *testing.T) {
	for _, kind := range []string{"nil", "input-id", "source-id", "source-nonconstant", "inventory", "parameter-id"} {
		t.Run(kind, func(t *testing.T) {
			_, p, _ := forwardCompositionSources(t)
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{0, 0}}}, Outputs: []int{0}}
			target := &tape
			input := 0
			var bindings map[string]train.ResidentTapeBinding
			switch kind {
			case "nil":
				target = nil
			case "input-id":
				input = -1
			case "source-id":
				p.inputValue = len(p.Tape.Tensors)
			case "source-nonconstant":
				p.inputValue = p.Tape.Outputs[0]
			case "inventory":
				bindings = map[string]train.ResidentTapeBinding{}
			case "parameter-id":
				bindings = map[string]train.ResidentTapeBinding{}
				for name := range p.ResidentBindings {
					bindings[name] = train.ResidentTapeBinding{Parameter: 1, Gradient: 0}
				}
			}
			if _, _, err := appendAdmittedFlowGraph(target, p, input, bindings); err == nil {
				t.Fatal("bad edge", kind)
			}
		})
	}
}

func TestTrainingCompositionMetadataCeilings(t *testing.T) {
	t.Run("distill", func(t *testing.T) {
		ep, pp := distillCompositionSources(t)
		for len(ep.Tape.Tensors)+len(pp.Tape.Tensors) <= 4096 {
			compositionConstant(&ep.Tape, 1)
		}
		if err := train.ValidateResidentTapeBindings(ep.Tape, ep.ResidentBindings, 64<<20); err != nil {
			t.Fatal(err)
		}
		if _, err := composeDistillTrainingTapes(ep, pp, 64<<20); err == nil || !strings.Contains(err.Error(), "combined graph ceiling") {
			t.Fatal("unbounded metadata", err)
		}
	})
	t.Run("forward", func(t *testing.T) {
		interpolation, p, e := forwardCompositionSources(t)
		for len(interpolation.Tensors)+len(p.Tape.Tensors)+len(e.Tape.Tensors)+2 <= 4096 {
			compositionConstant(&interpolation, 1)
		}
		if err := interpolation.Validate(64 << 20); err != nil {
			t.Fatal(err)
		}
		if _, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20); err == nil || !strings.Contains(err.Error(), "combined graph ceiling") {
			t.Fatal("unbounded metadata", err)
		}
	})
}
