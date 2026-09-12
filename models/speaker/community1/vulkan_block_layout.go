package community1

import (
	"context"
	"fmt"
	"math"
)

type vulkanBlockTensor struct {
	name  string
	shape []int
	data  []float32
}

type vulkanBlockStep struct {
	op             string
	out, x, a, b   string
	kernel, stride int
	padding        int
	relu           bool
}

type vulkanBlockLayout struct {
	input, output CHWShape
	tensors       []vulkanBlockTensor
	steps         []vulkanBlockStep
	weightBytes   uint64
	scratchBytes  uint64
}

// describeVulkanBasicBlock performs complete model/shape/coefficient preflight
// before any Vulkan call. It retains no source slices in the returned layout.
func describeVulkanBasicBlock(ctx context.Context, block *WeSpeakerBasicBlock, input CHWShape) (*vulkanBlockLayout, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if block == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: nil source")
	}
	output, err := block.OutputShape(input)
	if err != nil {
		return nil, err
	}
	l := &vulkanBlockLayout{input: input, output: output}
	add := func(name string, data []float32, shape ...int) {
		if err != nil {
			return
		}
		n := 1
		for _, dimension := range shape {
			if dimension < 1 || n > int(^uint(0)>>1)/dimension {
				err = fmt.Errorf("Community-1 Vulkan block: %s shape overflow", name)
				return
			}
			n *= dimension
		}
		if data != nil && len(data) != n {
			err = fmt.Errorf("Community-1 Vulkan block: %s length %d, want %d", name, len(data), n)
			return
		}
		owned := data
		if data != nil {
			owned = make([]float32, len(data))
			for start := 0; start < len(data); start += 4096 {
				if e := ctx.Err(); e != nil {
					err = e
					return
				}
				end := min(start+4096, len(data))
				for i, value := range data[start:end] {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						err = fmt.Errorf("Community-1 Vulkan block: nonfinite %s[%d]", name, start+i)
						return
					}
				}
				copy(owned[start:end], data[start:end])
			}
			l.weightBytes += uint64(len(data)) * 4
		} else {
			l.scratchBytes += uint64(n) * 4
		}
		l.tensors = append(l.tensors, vulkanBlockTensor{name: name, shape: append([]int(nil), shape...), data: owned})
	}
	prepareBN := func(prefix string, bn WeSpeakerBN, channels int) {
		if err != nil {
			return
		}
		scale, shift, prepareErr := prepareVulkanBN(ctx, bn, channels)
		if prepareErr != nil {
			err = fmt.Errorf("Community-1 Vulkan block: %s: %w", prefix, prepareErr)
			return
		}
		add(prefix+".scale", scale, channels)
		add(prefix+".shift", shift, channels)
	}
	add("conv1.weight", block.weights.Conv1, output.Channels, input.Channels, 3, 3)
	prepareBN("bn1", block.weights.BN1, output.Channels)
	add("conv2.weight", block.weights.Conv2, output.Channels, output.Channels, 3, 3)
	prepareBN("bn2", block.weights.BN2, output.Channels)
	if blockNeedsShortcut(block.cfg) {
		add("shortcut.weight", block.weights.Shortcut, output.Channels, input.Channels, 1, 1)
		prepareBN("shortcut.bn", block.weights.ShortcutBN, output.Channels)
	}
	identity, zero := make([]float32, output.Channels), make([]float32, output.Channels)
	for i := range identity {
		identity[i] = 1
	}
	add("relu.scale", identity, output.Channels)
	add("relu.shift", zero, output.Channels)
	add("input", nil, input.Channels, input.Frequency, input.Frames)
	add("a", nil, output.Channels, output.Frequency, output.Frames)
	add("b", nil, output.Channels, output.Frequency, output.Frames)
	if err != nil {
		return nil, err
	}
	l.steps = append(l.steps,
		vulkanBlockStep{op: "conv", out: "a", x: "input", a: "conv1.weight", kernel: 3, stride: block.cfg.Stride, padding: 1},
		vulkanBlockStep{op: "affine", out: "a", x: "a", a: "bn1.scale", b: "bn1.shift", relu: true},
		vulkanBlockStep{op: "conv", out: "b", x: "a", a: "conv2.weight", kernel: 3, stride: 1, padding: 1},
		vulkanBlockStep{op: "affine", out: "b", x: "b", a: "bn2.scale", b: "bn2.shift"},
	)
	shortcut := "input"
	if blockNeedsShortcut(block.cfg) {
		l.steps = append(l.steps,
			vulkanBlockStep{op: "conv", out: "a", x: "input", a: "shortcut.weight", kernel: 1, stride: block.cfg.Stride},
			vulkanBlockStep{op: "affine", out: "a", x: "a", a: "shortcut.bn.scale", b: "shortcut.bn.shift"},
		)
		shortcut = "a"
	}
	l.steps = append(l.steps,
		vulkanBlockStep{op: "add", out: "b", x: "b", a: shortcut},
		vulkanBlockStep{op: "affine", out: "b", x: "b", a: "relu.scale", b: "relu.shift", relu: true},
	)
	return l, ctx.Err()
}

func prepareVulkanBN(ctx context.Context, bn WeSpeakerBN, channels int) ([]float32, []float32, error) {
	if len(bn.Weight) != channels || len(bn.Bias) != channels || len(bn.RunningMean) != channels || len(bn.RunningVariance) != channels {
		return nil, nil, fmt.Errorf("invalid BatchNorm")
	}
	scale, shift := make([]float32, channels), make([]float32, channels)
	for i := range scale {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		variance := bn.RunningVariance[i]
		if variance < 0 {
			return nil, nil, fmt.Errorf("negative variance")
		}
		inv := float32(1 / math.Sqrt(float64(variance)+1e-5))
		scale[i] = inv * bn.Weight[i]
		shift[i] = bn.Bias[i] - bn.RunningMean[i]*scale[i]
		if math.IsNaN(float64(scale[i])) || math.IsInf(float64(scale[i]), 0) || math.IsNaN(float64(shift[i])) || math.IsInf(float64(shift[i]), 0) {
			return nil, nil, fmt.Errorf("nonfinite prepared coefficient")
		}
	}
	return scale, shift, nil
}

func vulkanBlockArenaBytes(tensors []vulkanBlockTensor, alignment uint64) (uint64, error) {
	if alignment < 4 {
		alignment = 4
	}
	if alignment&(alignment-1) != 0 {
		return 0, fmt.Errorf("Community-1 Vulkan block: invalid arena alignment")
	}
	var size uint64
	for _, tensor := range tensors {
		n := uint64(4)
		for _, dimension := range tensor.shape {
			if dimension < 1 || n > math.MaxUint64/uint64(dimension) {
				return 0, fmt.Errorf("Community-1 Vulkan block: arena size overflow")
			}
			n *= uint64(dimension)
		}
		if size > math.MaxUint64-(alignment-1) {
			return 0, fmt.Errorf("Community-1 Vulkan block: arena alignment overflow")
		}
		size = (size + alignment - 1) &^ (alignment - 1)
		if n > math.MaxUint64-size {
			return 0, fmt.Errorf("Community-1 Vulkan block: arena extent overflow")
		}
		size += n
	}
	return size, nil
}
