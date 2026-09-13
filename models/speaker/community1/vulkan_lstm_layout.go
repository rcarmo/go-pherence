package community1

import (
	"context"
	"fmt"
	"math"
)

type vulkanLSTMStep struct {
	output, input                      string
	weightIH, weightHH, biasIH, biasHH string
	hidden, cell                       string
	outputOffset                       int
	reverse                            bool
}

type vulkanLSTMLayout struct {
	cfg                           LSTMConfig
	frames, directions, stateSize int
	weights, scratch              []vulkanBlockTensor
	plans                         [][]vulkanLSTMStep
	hiddenNames, cellNames        []string
	inputName, outputName         string
	weightBytes, scratchBytes     uint64
}

func describeVulkanLSTM(ctx context.Context, model *LSTM, frames int) (*vulkanLSTMLayout, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: nil source")
	}
	if err := checkLSTMConfig(model.cfg); err != nil {
		return nil, err
	}
	if frames < 1 || frames > 4096 || len(model.layers) != model.cfg.NumLayers {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: invalid frames/layers")
	}
	directions, hidden := lstmDirections(model.cfg), model.cfg.HiddenSize
	layout := &vulkanLSTMLayout{cfg: model.cfg, frames: frames, directions: directions, stateSize: model.cfg.NumLayers * directions * hidden, inputName: "input"}
	add := func(dst *[]vulkanBlockTensor, name string, data []float32, shape ...int) error {
		n := 1
		for _, dimension := range shape {
			if dimension < 1 || n > int(^uint(0)>>1)/dimension {
				return fmt.Errorf("Community-1 Vulkan LSTM: %s shape overflow", name)
			}
			n *= dimension
		}
		if data != nil && len(data) != n {
			return fmt.Errorf("Community-1 Vulkan LSTM: %s length %d, want %d", name, len(data), n)
		}
		owned := data
		if data != nil {
			owned = make([]float32, len(data))
			for start := 0; start < len(data); start += 4096 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+4096, len(data))
				for i, value := range data[start:end] {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						return fmt.Errorf("Community-1 Vulkan LSTM: nonfinite %s[%d]", name, start+i)
					}
				}
				copy(owned[start:end], data[start:end])
			}
			layout.weightBytes += uint64(len(data)) * 4
		} else {
			layout.scratchBytes += uint64(n) * 4
		}
		*dst = append(*dst, vulkanBlockTensor{name: name, shape: append([]int(nil), shape...), data: owned})
		return nil
	}
	if err := add(&layout.scratch, layout.inputName, nil, frames, model.cfg.InputSize); err != nil {
		return nil, err
	}
	for layerIndex, layer := range model.layers {
		inputDim := model.cfg.InputSize
		if layerIndex > 0 {
			inputDim = directions * hidden
		}
		outputName := fmt.Sprintf("layer%d.output", layerIndex)
		if err := add(&layout.scratch, outputName, nil, frames, directions*hidden); err != nil {
			return nil, err
		}
		plan := make([]vulkanLSTMStep, 0, directions)
		for direction := 0; direction < directions; direction++ {
			weights := layer.Forward
			if direction == 1 {
				weights = layer.Reverse
			}
			prefix := fmt.Sprintf("layer%d.direction%d.", layerIndex, direction)
			for _, spec := range []struct {
				name  string
				data  []float32
				shape []int
			}{
				{"weight_ih", weights.WeightIH, []int{4 * hidden, inputDim}},
				{"weight_hh", weights.WeightHH, []int{4 * hidden, hidden}},
				{"bias_ih", weights.BiasIH, []int{4 * hidden}},
				{"bias_hh", weights.BiasHH, []int{4 * hidden}},
			} {
				if err := add(&layout.weights, prefix+spec.name, spec.data, spec.shape...); err != nil {
					return nil, err
				}
			}
			hiddenName, cellName := prefix+"hidden", prefix+"cell"
			if err := add(&layout.scratch, hiddenName, nil, hidden); err != nil {
				return nil, err
			}
			if err := add(&layout.scratch, cellName, nil, hidden); err != nil {
				return nil, err
			}
			layout.hiddenNames = append(layout.hiddenNames, hiddenName)
			layout.cellNames = append(layout.cellNames, cellName)
			inputName := layout.inputName
			if layerIndex > 0 {
				inputName = fmt.Sprintf("layer%d.output", layerIndex-1)
			}
			plan = append(plan, vulkanLSTMStep{output: outputName, input: inputName, weightIH: prefix + "weight_ih", weightHH: prefix + "weight_hh", biasIH: prefix + "bias_ih", biasHH: prefix + "bias_hh", hidden: hiddenName, cell: cellName, outputOffset: direction * hidden, reverse: direction == 1})
		}
		layout.plans = append(layout.plans, plan)
		layout.outputName = outputName
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return layout, nil
}
