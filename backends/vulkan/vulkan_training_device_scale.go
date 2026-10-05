package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

func (op *VkTrainingF32) AdamWScaledStage(ctx context.Context, parameter, gradient, m, v, ema, clip *VkTensorF32, p train.AdamWStep) (VkF32Stage, error) {
	if err := p.Validate(); err != nil {
		return VkF32Stage{}, err
	}
	if p.GradientScale != 1 {
		return VkF32Stage{}, fmt.Errorf("device-scale AdamW requires unit host scale")
	}
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.adamScaled == nil || parameter == nil || parameter.rank != 1 || parameter.shape[0] < 1 || parameter.shape[0] > 1<<26 {
		return VkF32Stage{}, fmt.Errorf("device-scale AdamW operator/rank/extent")
	}
	n := parameter.shape[0]
	tensors := []*VkTensorF32{parameter, gradient, m, v, ema, clip}
	for i, t := range tensors {
		size := n
		if i == 5 {
			size = 4
		}
		if t == nil || t.rank != 1 || t.shape[0] != size {
			return VkF32Stage{}, fmt.Errorf("device-scale AdamW buffer shape")
		}
		// Include read-only gradient/clip in disjoint admission as legacy AdamW does.
		binding, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		for j := 0; j < i; j++ {
			other, e := tensors[j].bindingLocked()
			if e != nil {
				return VkF32Stage{}, e
			}
			if vkBindingsOverlap(binding, other) {
				return VkF32Stage{}, fmt.Errorf("device-scale AdamW buffer alias")
			}
		}
	}
	words := []uint32{uint32(n), math.Float32bits(p.LearningRate), math.Float32bits(p.Beta1), math.Float32bits(p.Beta2), math.Float32bits(p.Epsilon), math.Float32bits(p.WeightDecay), math.Float32bits(p.Bias1), math.Float32bits(p.Bias2), math.Float32bits(p.EMADecay), math.Float32bits(1)}
	return trainingStageLocked(op.adamScaled, tensors, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4, uint64(n) * 4, uint64(n) * 4, 16}, []int{0, 2, 3, 4}, uint32(n), words)
}
func (d *VulkanResidentDriver) scaledStage(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) (VkF32Stage, error) {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return VkF32Stage{}, fmt.Errorf("nil/terminal device-scale Vulkan driver")
	}
	b, err := vkResidentBuffers(p, g, m, v, ema, clip)
	if err != nil {
		return VkF32Stage{}, err
	}
	if b[0].rank != 1 || b[0].shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("device-scale Vulkan update extent")
	}
	return d.Operator.AdamWScaledStage(ctx, b[0], b[1], b[2], b[3], b[4], b[5], step)
}
func (d *VulkanResidentDriver) ValidateScaledUpdate(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	_, err := d.scaledStage(ctx, p, g, m, v, ema, clip, n, step)
	return err
}
func (d *VulkanResidentDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	s, err := d.scaledStage(ctx, p, g, m, v, ema, clip, n, step)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}
