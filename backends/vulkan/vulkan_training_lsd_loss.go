package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *VkTrainingF32) LSDLossStage(ctx context.Context, dst, prediction, desired, logvar *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.lsdLoss == nil || prediction == nil || prediction.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("training LSD loss owner/rank")
	}
	c := prediction.shape[0]
	n, err := train.LSDLossShape(c, mode)
	if err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{prediction, desired, logvar, dst}
	sizes := []int{c, c, 1, n}
	bytes := make([]uint64, 4)
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != sizes[i] {
			return VkF32Stage{}, fmt.Errorf("training LSD loss extent")
		}
		bytes[i] = uint64(sizes[i]) * 4
	}
	return trainingStageLocked(op.lsdLoss, tensors, bytes, []int{3}, uint32(n), []uint32{uint32(c), mode})
}
