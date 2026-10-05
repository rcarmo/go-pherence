package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

// WeightTrainingTape covers the learned affine/ReLU log-variance MLP and its
// input/parameter VJP. Its graph uses the same immutable binding and disjoint
// overwrite contract as transformer tapes. Parameters stay fixed through paired
// forward/backward evaluations. Scalar losses are outside this graph.
type WeightTrainingTape struct {
	Tape             train.Tape
	ParameterNames   []string
	ResidentBindings map[string]train.ResidentTapeBinding
	sampleSeedIDs    [2]int // explicit composition edges, not public output changes
}
type WeightTrainingResult struct {
	Output, DInput []float32
	Parameters     map[string][]float32
}

func (m *LSDWeightMLP) BuildWeightTrainingTape(input, seed []float32, rows int, maxBytes int64) (WeightTrainingTape, error) {
	return m.buildWeightTrainingTape(input, seed, rows, maxBytes, nil, false)
}
func (m *LSDWeightMLP) buildWeightTrainingTape(input, seed []float32, rows int, maxBytes int64, samples *WeightSampleInputs, borrowParameters bool) (WeightTrainingTape, error) {
	var result WeightTrainingTape
	if m == nil || rows < 1 || rows > 1024 || maxBytes <= 0 || len(m.Layers) == 0 || len(m.Layers) > 16 || m.Layers[0].In != 2 || m.Layers[len(m.Layers)-1].Out != 1 {
		return result, fmt.Errorf("invalid weighting tape input/ceilings")
	}
	if samples == nil {
		if len(input) != rows*2 || len(seed) != rows || !finiteF32(input) || !finiteF32(seed) {
			return result, fmt.Errorf("invalid weighting tape input/ceilings")
		}
	} else if err := samples.validate(rows); err != nil {
		return result, err
	}
	for i, l := range m.Layers {
		if err := validateOwnedTrainingLinear(l, true); err != nil {
			return result, err
		}
		if err := train.LinearShape(rows, l.In, l.Out); err != nil {
			return result, err
		}
		if i > 0 && l.In != m.Layers[i-1].Out {
			return result, fmt.Errorf("weighting tape layer topology")
		}
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	var x, dy int
	if samples == nil {
		x = g.constant([]int{rows, 2}, input)
	} else {
		x, dy, result.sampleSeedIDs = samples.build(g)
	}
	current := x
	inputs, pre, weights, biases := make([]int, len(m.Layers)), make([]int, len(m.Layers)), make([]int, len(m.Layers)), make([]int, len(m.Layers))
	for i, l := range m.Layers {
		prefix := fmt.Sprintf("flow.w_s_t.%d", i*2)
		inputs[i] = current
		weights[i] = g.param(prefix+".weight", []int{l.Out, l.In}, l.Weight)
		biases[i] = g.param(prefix+".bias", []int{l.Out}, l.Bias)
		pre[i] = g.command("linear-forward", []int{current, g.zero(rows, l.In), weights[i], biases[i]}, []int{rows, l.Out}, 0)
		current = pre[i]
		if i < len(m.Layers)-1 {
			current = g.reshape(g.command("relu", []int{g.flat(current), g.zero(rows * l.Out)}, []int{rows * l.Out}, 0), rows, l.Out)
		}
	}
	output := g.flat(current)
	if samples == nil {
		dy = g.constant([]int{rows, 1}, seed)
	}
	gradients := map[int]int{}
	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := m.Layers[i]
		if i < len(m.Layers)-1 {
			dy = g.reshape(g.command("relu", []int{g.flat(pre[i]), g.flat(dy)}, []int{rows * l.Out}, 1), rows, l.Out)
		}
		operands := []int{inputs[i], g.zero(rows, l.In), weights[i], dy, g.zero(rows, l.Out)}
		gradients[weights[i]] = g.command("linear-backward", operands, []int{l.Out, l.In}, 2)
		gradients[biases[i]] = g.command("linear-backward", operands, []int{l.Out}, 3)
		dy = g.command("linear-backward", operands, []int{rows, l.In}, 0)
	}
	g.tape.Outputs = []int{output, g.flat(dy)}
	result.ResidentBindings = map[string]train.ResidentTapeBinding{}
	for name := range g.params {
		result.ParameterNames = append(result.ParameterNames, name)
	}
	sort.Strings(result.ParameterNames)
	for _, name := range result.ParameterNames {
		id := g.flat(gradients[g.params[name]])
		g.tape.Outputs = append(g.tape.Outputs, id)
		result.ResidentBindings[name] = train.ResidentTapeBinding{Parameter: g.params[name], Gradient: id}
	}
	// Charge complete logical graph before copying any tensor values/zeros.
	var bytes int64
	if len(g.tape.Tensors) > 4096 || len(g.tape.Commands) > 4096 {
		return result, fmt.Errorf("weighting tape graph ceiling")
	}
	for id, s := range g.tape.Tensors {
		n, err := s.Elements()
		if err != nil {
			return result, err
		}
		bytes += int64(n) * 4
		if bytes > maxBytes {
			return result, fmt.Errorf("weighting tape byte ceiling")
		}
		if x, ok := g.constants[id]; ok && (len(x) != n || !finiteF32(x)) {
			return result, fmt.Errorf("weighting tape constant shape/nonfinite")
		}
	}
	for id, s := range g.tape.Tensors {
		if x, ok := g.constants[id]; ok {
			g.tape.Tensors[id].Values = g.constantValues(id, x)
		} else if g.zeros[id] {
			n, _ := s.Elements()
			g.tape.Tensors[id].Values = g.zeroValues(n)
		}
	}
	if err := g.tape.Validate(maxBytes); err != nil {
		return result, err
	}
	result.Tape = g.tape
	return result, nil
}
func (t WeightTrainingTape) Decode(values [][]float32) (WeightTrainingResult, error) {
	var result WeightTrainingResult
	if len(values) != 2+len(t.ParameterNames) || len(values) != len(t.Tape.Outputs) {
		return result, fmt.Errorf("weighting tape result count")
	}
	for i, x := range values {
		id := t.Tape.Outputs[i]
		if id < 0 || id >= len(t.Tape.Tensors) {
			return result, fmt.Errorf("weighting tape output id")
		}
		n, err := t.Tape.Tensors[id].Elements()
		if err != nil || len(x) != n || !finiteF32(x) {
			return result, fmt.Errorf("weighting tape result shape/nonfinite")
		}
	}
	result.Output, result.DInput = append([]float32(nil), values[0]...), append([]float32(nil), values[1]...)
	result.Parameters = map[string][]float32{}
	for i, name := range t.ParameterNames {
		result.Parameters[name] = append([]float32(nil), values[i+2]...)
	}
	return result, nil
}
func validateWeightBoundary(r WeightTrainingResult, rows int, inputs ...[]float32) error {
	if len(r.Output) != rows || len(r.DInput) != rows*2 || !finiteF32(r.Output) || !finiteF32(r.DInput) || len(r.Parameters) != 0 {
		return fmt.Errorf("resident weighting boundary shape/nonfinite/host gradients")
	}
	if f32SlicesOverlap(r.Output, r.DInput) {
		return fmt.Errorf("weighting outputs overlap")
	}
	for _, out := range [][]float32{r.Output, r.DInput} {
		for _, in := range inputs {
			if f32SlicesOverlap(out, in) {
				return fmt.Errorf("weighting backend borrowed input")
			}
		}
	}
	return nil
}

// TrainingWeightBackend offloads the complete learned weighting MLP, keeping
// parameter gradients resident. The boolean must be false for forward and
// true exactly once for the seeded VJP; no implicit accumulation is inferred.
type TrainingWeightBackend interface {
	EvaluateWeight(*LSDWeightMLP, []float32, []float32, int, bool) (WeightTrainingResult, error)
}
