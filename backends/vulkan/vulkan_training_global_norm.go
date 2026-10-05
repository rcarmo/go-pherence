package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

func (d *VulkanResidentDriver) globalNormStage(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32, maxNorm float32) (VkF32Stage, error) {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return VkF32Stage{}, fmt.Errorf("nil/terminal Vulkan global norm driver")
	}
	count, out := 4, 4
	if mode != 1 {
		var err error
		count, err = train.NormMergeShape(n, previous)
		if err != nil {
			return VkF32Stage{}, err
		}
	} else {
		out = 4
		if err := train.NormClipShape(previous, maxNorm); err != nil {
			return VkF32Stage{}, err
		}
	}
	values, err := vkResidentBuffers(a, b, dst)
	if err != nil {
		return VkF32Stage{}, err
	}
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if d.Operator.globalNorm == nil {
		return VkF32Stage{}, fmt.Errorf("global norm kernel unavailable")
	}
	for i, t := range values {
		size := []int{4, count, out}[i]
		if t.rank != 1 || t.shape[0] != size {
			return VkF32Stage{}, fmt.Errorf("global norm tensor shape")
		}
	}
	return trainingStageLocked(d.Operator.globalNorm, values, []uint64{16, uint64(count) * 4, uint64(out) * 4}, []int{2}, 1, []uint32{uint32(n), mode, math.Float32bits(float32(previous)), math.Float32bits(maxNorm)})
}
func (d *VulkanResidentDriver) ValidateNormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	if mode != 0 && mode != 2 {
		return fmt.Errorf("global norm merge mode")
	}
	_, err := d.globalNormStage(ctx, dst, a, b, n, previous, mode, 1)
	return err
}
func (d *VulkanResidentDriver) ValidateNormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	_, err := d.globalNormStage(ctx, dst, a, a, 1, elements, 1, maxNorm)
	return err
}
func (d *VulkanResidentDriver) NormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	if mode != 0 && mode != 2 {
		return fmt.Errorf("global norm merge mode")
	}
	s, err := d.globalNormStage(ctx, dst, a, b, n, previous, mode, 1)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}
func (d *VulkanResidentDriver) NormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	s, err := d.globalNormStage(ctx, dst, a, a, 1, elements, 1, maxNorm)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}
