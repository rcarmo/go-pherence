package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// ProjectionTrainingTape evaluates an owned affine projection and its VJP.
// Parameter names are explicit because speaker_proj_weight has no .weight
// suffix. Forward/backward tensors are contiguous row-major; bias is optional.
type ProjectionTrainingTape struct {
	Tape             train.Tape
	ResidentBindings map[string]train.ResidentTapeBinding
}
type ProjectionTrainingResult struct{ Output, DInput []float32 }

func BuildProjectionTrainingTape(linear LinearF32, weightName, biasName string, input, seed []float32, rows int, maxBytes int64) (ProjectionTrainingTape, error) {
	return buildProjectionTrainingTape(linear, weightName, biasName, input, seed, rows, maxBytes, false)
}
func buildProjectionTrainingTape(linear LinearF32, weightName, biasName string, input, seed []float32, rows int, maxBytes int64, borrowParameters bool) (ProjectionTrainingTape, error) {
	var result ProjectionTrainingTape
	if err := train.LinearShape(rows, linear.In, linear.Out); err != nil {
		return result, err
	}
	if maxBytes <= 0 || weightName == "" || len(weightName) > 512 || (linear.Bias != nil && (biasName == "" || biasName == weightName || len(biasName) > 512)) || len(input) != rows*linear.In || len(seed) != rows*linear.Out || !finiteF32(input) || !finiteF32(seed) {
		return result, fmt.Errorf("invalid projection tape inputs/names/ceiling")
	}
	if err := validateOwnedTrainingLinear(linear, linear.Bias != nil); err != nil {
		return result, err
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	x := g.constant([]int{rows, linear.In}, input)
	dy := g.constant([]int{rows, linear.Out}, seed)
	w := g.param(weightName, []int{linear.Out, linear.In}, linear.Weight)
	bias := g.zero(linear.Out)
	if linear.Bias != nil {
		bias = g.param(biasName, []int{linear.Out}, linear.Bias)
	}
	xt := g.zero(rows, linear.In)
	yt := g.zero(rows, linear.Out)
	y := g.command("linear-forward", []int{x, xt, w, bias}, []int{rows, linear.Out}, 0)
	operands := []int{x, xt, w, dy, yt}
	dx := g.command("linear-backward", operands, []int{rows, linear.In}, 0)
	dw := g.command("linear-backward", operands, []int{linear.Out, linear.In}, 2)
	result.ResidentBindings = map[string]train.ResidentTapeBinding{weightName: {Parameter: w, Gradient: dw}}
	g.tape.Outputs = []int{y, dx, dw}
	if linear.Bias != nil {
		db := g.command("linear-backward", operands, []int{linear.Out}, 3)
		result.ResidentBindings[biasName] = train.ResidentTapeBinding{Parameter: bias, Gradient: db}
		g.tape.Outputs = append(g.tape.Outputs, db)
	}
	var bytes int64
	for id, s := range g.tape.Tensors {
		n, err := s.Elements()
		if err != nil {
			return result, err
		}
		bytes += int64(n) * 4
		if bytes > maxBytes {
			return result, fmt.Errorf("projection tape byte ceiling")
		}
		if v, ok := g.constants[id]; ok && (len(v) != n || !finiteF32(v)) {
			return result, fmt.Errorf("projection tape constant shape/nonfinite")
		}
	}
	for id, s := range g.tape.Tensors {
		if v, ok := g.constants[id]; ok {
			g.tape.Tensors[id].Values = g.constantValues(id, v)
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

// TrainingProjectionBackend keeps projection parameter gradients resident.
// Projection is one of speaker/input/EOS, selected explicitly by the caller.
// Forward never accumulates; seeded backward accumulates once. Parameters must
// remain immutable between paired calls; nil backend preserves CPU defaults.
type TrainingProjectionBackend interface {
	EvaluateProjection(*FlowLMTrainingCPU, string, []float32, []float32, int, bool) (ProjectionTrainingResult, error)
}

func validateProjectionResult(r ProjectionTrainingResult, rows, in, out int, inputs ...[]float32) error {
	if len(r.Output) != rows*out || len(r.DInput) != rows*in || !finiteF32(r.Output) || !finiteF32(r.DInput) {
		return fmt.Errorf("projection boundary shape/nonfinite")
	}
	if f32SlicesOverlap(r.Output, r.DInput) {
		return fmt.Errorf("projection boundary outputs overlap")
	}
	for _, x := range [][]float32{r.Output, r.DInput} {
		for _, v := range inputs {
			if f32SlicesOverlap(x, v) {
				return fmt.Errorf("projection boundary borrowed input")
			}
		}
	}
	return nil
}
