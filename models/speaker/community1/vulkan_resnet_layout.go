package community1

import (
	"context"
	"fmt"
	"math"
)

type vulkanResNetLayout struct {
	frames        int
	input, output CHWShape
	weights       [][]vulkanBlockTensor
	scratch       []vulkanBlockTensor
	plans         [][]vulkanBlockStep
	outputName    string
	weightBytes   uint64
	scratchBytes  uint64
}

// describeVulkanResNetTrunk validates and copies the complete fixed-frame CNN
// before any Vulkan call. Pooling and embedding projection remain host-owned.
func describeVulkanResNetTrunk(ctx context.Context, model *WeSpeakerResNet34, frames int) (*vulkanResNetLayout, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: nil source")
	}
	final, err := model.FrameShape(frames)
	if err != nil {
		return nil, err
	}
	l := &vulkanResNetLayout{frames: frames, input: CHWShape{1, model.cfg.MelBins, frames}, output: final}
	stemScale, stemShift, err := prepareVulkanBN(ctx, model.stemBN, model.cfg.BaseChannels)
	if err != nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk stem: %w", err)
	}
	stemWeight := append([]float32(nil), model.stem...)
	for i, value := range stemWeight {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("Community-1 Vulkan trunk: nonfinite stem[%d]", i)
		}
	}
	stemWeights := []vulkanBlockTensor{
		{name: "stem.conv.weight", shape: []int{model.cfg.BaseChannels, 1, 3, 3}, data: stemWeight},
		{name: "stem.bn.scale", shape: []int{model.cfg.BaseChannels}, data: stemScale},
		{name: "stem.bn.shift", shape: []int{model.cfg.BaseChannels}, data: stemShift},
	}
	for _, tensor := range stemWeights {
		l.weightBytes += uint64(len(tensor.data)) * 4
	}
	l.weights = append(l.weights, stemWeights)
	l.scratch = append(l.scratch, vulkanBlockTensor{name: "input", shape: []int{1, model.cfg.MelBins, frames}})
	l.scratchBytes = uint64(model.cfg.MelBins*frames) * 4

	var slots [4][3]string
	shape := CHWShape{model.cfg.BaseChannels, model.cfg.MelBins, frames}
	for stage := range slots {
		for slot := range slots[stage] {
			name := fmt.Sprintf("stage%d.%d", stage, slot)
			slots[stage][slot] = name
			l.scratch = append(l.scratch, vulkanBlockTensor{name: name, shape: []int{shape.Channels, shape.Frequency, shape.Frames}})
			l.scratchBytes += uint64(shape.Channels*shape.Frequency*shape.Frames) * 4
		}
		if stage < len(slots)-1 {
			shape = CHWShape{shape.Channels * 2, (shape.Frequency + 1) / 2, (shape.Frames + 1) / 2}
		}
	}
	l.plans = append(l.plans, []vulkanBlockStep{
		{op: "conv", out: slots[0][0], x: "input", a: "stem.conv.weight", kernel: 3, stride: 1, padding: 1},
		{op: "affine", out: slots[0][0], x: slots[0][0], a: "stem.bn.scale", b: "stem.bn.shift", relu: true},
	})
	current := slots[0][0]
	inputShape := CHWShape{model.cfg.BaseChannels, model.cfg.MelBins, frames}
	for stage, blocks := range model.stages {
		for index, block := range blocks {
			blockLayout, err := describeVulkanBasicBlock(ctx, block, inputShape)
			if err != nil {
				return nil, fmt.Errorf("Community-1 Vulkan trunk stage%d block%d: %w", stage, index, err)
			}
			prefix := fmt.Sprintf("stage%d.block%d.", stage, index)
			weights := make([]vulkanBlockTensor, 0, len(blockLayout.tensors)-3)
			for _, tensor := range blockLayout.tensors {
				if tensor.data == nil {
					continue
				}
				tensor.name = prefix + tensor.name
				weights = append(weights, tensor)
				l.weightBytes += uint64(len(tensor.data)) * 4
			}
			l.weights = append(l.weights, weights)
			available := make([]string, 0, 3)
			for _, name := range slots[stage] {
				if name != current {
					available = append(available, name)
				}
			}
			if blockNeedsShortcut(block.cfg) {
				// A stage transition reads the previous resolution, so all three
				// buffers at the new resolution are available.
				available = append([]string(nil), slots[stage][:]...)
			}
			if len(available) < 2 || (blockNeedsShortcut(block.cfg) && len(available) < 3) {
				return nil, fmt.Errorf("Community-1 Vulkan trunk: scratch assignment")
			}
			a, b := available[0], available[1]
			plan := make([]vulkanBlockStep, 0, len(blockLayout.steps))
			plan = append(plan,
				vulkanBlockStep{op: "conv", out: a, x: current, a: prefix + "conv1.weight", kernel: 3, stride: block.cfg.Stride, padding: 1},
				vulkanBlockStep{op: "affine", out: a, x: a, a: prefix + "bn1.scale", b: prefix + "bn1.shift", relu: true},
				vulkanBlockStep{op: "conv", out: b, x: a, a: prefix + "conv2.weight", kernel: 3, stride: 1, padding: 1},
				vulkanBlockStep{op: "affine", out: b, x: b, a: prefix + "bn2.scale", b: prefix + "bn2.shift"},
			)
			shortcut := current
			if blockNeedsShortcut(block.cfg) {
				shortcut = available[2]
				plan = append(plan,
					vulkanBlockStep{op: "conv", out: shortcut, x: current, a: prefix + "shortcut.weight", kernel: 1, stride: block.cfg.Stride},
					vulkanBlockStep{op: "affine", out: shortcut, x: shortcut, a: prefix + "shortcut.bn.scale", b: prefix + "shortcut.bn.shift"},
				)
			}
			plan = append(plan,
				vulkanBlockStep{op: "add", out: b, x: b, a: shortcut},
				vulkanBlockStep{op: "affine", out: b, x: b, a: prefix + "relu.scale", b: prefix + "relu.shift", relu: true},
			)
			l.plans = append(l.plans, plan)
			current = b
			inputShape = blockLayout.output
		}
	}
	if inputShape != final {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: final shape mismatch")
	}
	l.outputName = current
	return l, ctx.Err()
}
