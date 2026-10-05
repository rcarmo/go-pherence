package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

func (op *VkTrainingF32) EOSStage(ctx context.Context, dst, x, labels *VkTensorF32, active int, mode uint32, weight float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.eos == nil || x == nil || x.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("training EOS owner/rank")
	}
	n := x.shape[0]
	out, err := train.EOSShape(n, active, mode, weight)
	if err != nil {
		return VkF32Stage{}, err
	}
	if dst == nil || labels == nil || dst.rank != 1 || labels.rank != 1 || labels.shape[0] != n || dst.shape[0] != out {
		return VkF32Stage{}, fmt.Errorf("training EOS shape")
	}
	return trainingStageLocked(op.eos, []*VkTensorF32{x, labels, dst}, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(out) * 4}, []int{2}, uint32(out), []uint32{uint32(n), uint32(active), mode, math.Float32bits(weight)})
}
