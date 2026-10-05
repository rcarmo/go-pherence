package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *VkTrainingF32) TimeEmbeddingStage(ctx context.Context, dst, frequency, time, gv, gt *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.timeEmbedding == nil || frequency == nil || frequency.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("training time embedding owner/rank")
	}
	half := frequency.shape[0]
	n, err := train.TimeEmbeddingShape(half, mode)
	if err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{frequency, time, gv, gt, dst}
	sizes := []int{half, 2, 2 * half, 2 * half, n}
	bytes := make([]uint64, len(sizes))
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != sizes[i] {
			return VkF32Stage{}, fmt.Errorf("training time embedding shape")
		}
		bytes[i] = uint64(sizes[i]) * 4
	}
	return trainingStageLocked(op.timeEmbedding, tensors, bytes, []int{4}, uint32(n), []uint32{uint32(half), mode})
}
