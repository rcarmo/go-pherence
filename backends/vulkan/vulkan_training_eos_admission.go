package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *VkTrainingF32) EOSAdmissionStage(ctx context.Context, hidden, weight, bias, dst *VkTensorF32, rows, width int) (VkF32Stage, error) {
	n, budget, err := train.EOSAdmissionShape(rows, width)
	if err != nil {
		return VkF32Stage{}, err
	}
	if err = vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.eosAdmission == nil {
		return VkF32Stage{}, fmt.Errorf("nil Vulkan EOS admission operator")
	}
	tensors := []*VkTensorF32{hidden, weight, bias, dst}
	extents := []int{n, width, 1, 1}
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != extents[i] {
			return VkF32Stage{}, fmt.Errorf("EOS admission extent")
		}
	}
	return trainingStageLocked(op.eosAdmission, tensors, []uint64{uint64(n) * 4, uint64(width) * 4, 4, 4}, []int{3}, 1, []uint32{uint32(rows), uint32(width), budget})
}
