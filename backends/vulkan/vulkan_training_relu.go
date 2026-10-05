package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// ReLUStage overwrites an owned forward/VJP vector; derivative at zero is zero.
func (op *VkTrainingF32) ReLUStage(ctx context.Context, dst, x, seed *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.relu == nil || x == nil || dst == nil || seed == nil || x.rank != 1 || dst.rank != 1 || seed.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid training ReLU rank/operator")
	}
	n := x.shape[0]
	if err := train.ReLUShape(n, mode); err != nil {
		return VkF32Stage{}, err
	}
	if dst.shape[0] != n || seed.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("training ReLU shape")
	}
	return trainingStageLocked(op.relu, []*VkTensorF32{x, seed, dst}, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(n), mode})
}
