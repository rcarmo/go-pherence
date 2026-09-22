package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// LSDWeightMLP implements upstream w_s_t: affine/ReLU hidden layers followed
// by one affine output. All linears own mutable F32 weights and biases.
type LSDWeightMLP struct {
	Layers []LinearF32
}

type LSDWeightGradients struct {
	Layers []LinearF32Gradient
}

type lsdWeightTape struct {
	inputs, preactivations [][]float32
}

func (m *LSDWeightMLP) forward(inputs []float32, rows int) (*lsdWeightTape, []float32, error) {
	inputElements, ok := checked.MulInt(rows, 2)
	if m == nil || rows <= 0 || !ok || len(m.Layers) == 0 || m.Layers[0].In != 2 || m.Layers[len(m.Layers)-1].Out != 1 || len(inputs) != inputElements {
		return nil, nil, fmt.Errorf("invalid Pocket TTS LSD weighting input")
	}
	for i, layer := range m.Layers {
		if err := validateOwnedTrainingLinear(layer, true); err != nil {
			return nil, nil, err
		}
		if i > 0 && layer.In != m.Layers[i-1].Out {
			return nil, nil, fmt.Errorf("invalid Pocket TTS LSD weighting topology")
		}
	}
	tape := &lsdWeightTape{inputs: make([][]float32, len(m.Layers)), preactivations: make([][]float32, len(m.Layers))}
	current := append([]float32(nil), inputs...)
	for index, layer := range m.Layers {
		tape.inputs[index] = append([]float32(nil), current...)
		preElements, ok := checked.MulInt(rows, layer.Out)
		if !ok {
			return nil, nil, fmt.Errorf("invalid Pocket TTS LSD weighting shape")
		}
		pre := make([]float32, preElements)
		for row := 0; row < rows; row++ {
			copy(pre[row*layer.Out:(row+1)*layer.Out], linearForwardTraining(layer, current[row*layer.In:(row+1)*layer.In]))
		}
		tape.preactivations[index] = pre
		current = append([]float32(nil), pre...)
		if index != len(m.Layers)-1 {
			for i := range current {
				if current[i] < 0 {
					current[i] = 0
				}
			}
		}
	}
	return tape, current, nil
}

func (m *LSDWeightMLP) backward(tape *lsdWeightTape, dOutput []float32, rows int) (*LSDWeightGradients, error) {
	if m == nil || tape == nil || len(tape.inputs) != len(m.Layers) || len(dOutput) != rows {
		return nil, fmt.Errorf("invalid Pocket TTS LSD weighting backward")
	}
	gradients := &LSDWeightGradients{Layers: make([]LinearF32Gradient, len(m.Layers))}
	current := append([]float32(nil), dOutput...)
	for index := len(m.Layers) - 1; index >= 0; index-- {
		layer := m.Layers[index]
		gradients.Layers[index] = newLinearGradient(layer)
		dInputElements, ok := checked.MulInt(rows, layer.In)
		if !ok {
			return nil, fmt.Errorf("invalid Pocket TTS LSD weighting backward shape")
		}
		dInput := make([]float32, dInputElements)
		for row := 0; row < rows; row++ {
			d := linearBackwardTraining(layer, tape.inputs[index][row*layer.In:(row+1)*layer.In], current[row*layer.Out:(row+1)*layer.Out], &gradients.Layers[index])
			copy(dInput[row*layer.In:(row+1)*layer.In], d)
		}
		if index > 0 {
			for i, value := range tape.preactivations[index-1] {
				if value <= 0 {
					dInput[i] = 0
				}
			}
		}
		current = dInput
	}
	return gradients, nil
}
