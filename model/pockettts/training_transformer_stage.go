package pockettts

import (
	"fmt"
	"sort"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// A complete, bounded source for one original transformer layer. This private
// builder is not an execution coordinator and never splits a rejected source.
// The caller must preflight the full model and keep all parameter owners live.
func buildTransformerLayerStage(m *TransformerCPU, index int, sequence, seed []float32, rows int, forwardOnly bool, maxBytes int64, pools ...*trainingZeroPool) (TransformerTrainingTape, error) {
	var result TransformerTrainingTape
	var pool *trainingZeroPool
	if len(pools) > 1 {
		return result, fmt.Errorf("layer placeholder pool count")
	}
	if len(pools) == 1 {
		pool = pools[0]
	}
	if m == nil || index < 0 || index >= len(m.Layers) || maxBytes <= 0 {
		return result, fmt.Errorf("transformer layer stage topology/ceiling")
	}
	if err := validateTransformerStageModel(m, sequence, seed, rows); err != nil {
		return result, err
	}
	stage := *m
	stage.Layers = m.Layers[index : index+1]
	stage.FinalWeight = nil
	stage.FinalBias = nil
	g := &transformerTapeGraph{borrowParameters: true, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, transientZeros: true, forwardOnly: forwardOnly, layerOffset: index}
	input := g.constant([]int{rows, m.Width}, sequence)
	dy := g.constant([]int{rows, m.Width}, seed)
	output, dx, gradients, err := stage.appendTransformerTrainingGraph(g, input, dy, rows)
	if err != nil {
		return result, err
	}
	result.ResidentBindings = map[string]train.ResidentTapeBinding{}
	g.tape.Outputs = []int{output}
	if !forwardOnly {
		g.tape.Outputs = append(g.tape.Outputs, dx)
	}
	for name := range g.params {
		result.ParameterNames = append(result.ParameterNames, name)
	}
	sort.Strings(result.ParameterNames)
	for _, name := range result.ParameterNames {
		id := g.params[name]
		gradient := 0
		if forwardOnly {
			n, _ := g.tape.Tensors[id].Elements()
			gradient, err = g.workspacePlaceholder(n, pool)
			if err != nil {
				return result, err
			}
		} else {
			var ok bool
			gradient, ok = gradients[id]
			if !ok {
				return result, fmt.Errorf("layer stage missing gradient %s", name)
			}
		}
		result.ResidentBindings[name] = train.ResidentTapeBinding{Parameter: id, Gradient: gradient}
		g.tape.Outputs = append(g.tape.Outputs, gradient)
	}
	built, err := materializeSequenceTape(g, result.ResidentBindings, maxBytes)
	if err != nil {
		return result, err
	}
	if err = train.ValidateResidentTapeBindings(built.Tape, result.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	result.Tape = built.Tape
	result.rows = rows
	result.width = m.Width
	return result, nil
}

// Stage selection never hides an invalid original model source.
func validateTransformerStageModel(m *TransformerCPU, sequence, seed []float32, rows int) error {
	if err := validateTrainableTransformer(m, sequence, seed, rows); err != nil {
		return err
	}
	if err := preflightTransformerTrainingTape(m, rows); err != nil {
		return err
	}
	original := map[string][]float32{}
	addTransformerParameters(original, m)
	for name, values := range original {
		if !finiteF32(values) {
			return fmt.Errorf("transformer stage full model nonfinite parameter %s", name)
		}
	}
	return nil
}

func buildTransformerFinalNormStage(m *TransformerCPU, sequence, seed []float32, rows int, forwardOnly bool, maxBytes int64, pools ...*trainingZeroPool) (TransformerTrainingTape, error) {
	var result TransformerTrainingTape
	var pool *trainingZeroPool
	if len(pools) > 1 {
		return result, fmt.Errorf("norm placeholder pool count")
	}
	if len(pools) == 1 {
		pool = pools[0]
	}
	if maxBytes <= 0 || m == nil || m.FinalWeight == nil {
		return result, fmt.Errorf("transformer final norm stage topology/ceiling")
	}
	if err := validateTransformerStageModel(m, sequence, seed, rows); err != nil {
		return result, err
	}
	g := &transformerTapeGraph{borrowParameters: true, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}, transientZeros: true}
	x := g.constant([]int{rows, m.Width}, sequence)
	dy := g.constant([]int{rows, m.Width}, seed)
	out, norm := g.norm("flow_lm.out_norm", x, rows, m.Width, m.FinalWeight, m.FinalBias)
	var dw, db int
	g.tape.Outputs = []int{out}
	if forwardOnly {
		var err error
		dw, err = g.workspacePlaceholder(m.Width, pool)
		if err != nil {
			return result, err
		}
		db, err = g.workspacePlaceholder(m.Width, pool)
		if err != nil {
			return result, err
		}
	} else {
		var dx int
		dx, dw, db = g.normBack(norm, dy, rows, m.Width)
		g.tape.Outputs = append(g.tape.Outputs, dx)
	}
	result.ParameterNames = []string{"flow_lm.out_norm.bias", "flow_lm.out_norm.weight"}
	result.ResidentBindings = map[string]train.ResidentTapeBinding{"flow_lm.out_norm.weight": {Parameter: norm.w, Gradient: dw}, "flow_lm.out_norm.bias": {Parameter: norm.b, Gradient: db}}
	for _, name := range result.ParameterNames {
		g.tape.Outputs = append(g.tape.Outputs, result.ResidentBindings[name].Gradient)
	}
	built, err := materializeSequenceTape(g, result.ResidentBindings, maxBytes)
	if err != nil {
		return result, err
	}
	if err = train.ValidateResidentTapeBindings(built.Tape, result.ResidentBindings, maxBytes); err != nil {
		return result, err
	}
	result.Tape = built.Tape
	result.rows = rows
	result.width = m.Width
	return result, nil
}
