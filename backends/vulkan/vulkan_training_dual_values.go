package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *VkTrainingF32) LinearDualForwardStage(ctx context.Context, dst, x, xt, w, bias *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.linearDualForward == nil || dst == nil || x == nil || xt == nil || w == nil || bias == nil || mode > 1 || x.rank != 2 || xt.rank != 2 || w.rank != 2 || bias.rank != 1 || dst.rank != 2 {
		return VkF32Stage{}, fmt.Errorf("dual linear forward rank/mode/operator")
	}
	m, k, n := x.shape[0], x.shape[1], w.shape[0]
	if err := train.LinearShape(m, k, n); err != nil {
		return VkF32Stage{}, err
	}
	if xt.shape[0] != m || xt.shape[1] != k || w.shape[1] != k || bias.shape[0] != n || dst.shape[0] != m || dst.shape[1] != n {
		return VkF32Stage{}, fmt.Errorf("dual linear forward shape mismatch")
	}
	return trainingStageLocked(op.linearDualForward, []*VkTensorF32{x, xt, w, bias, dst}, []uint64{uint64(m*k) * 4, uint64(m*k) * 4, uint64(n*k) * 4, uint64(n) * 4, uint64(m*n) * 4}, []int{4}, uint32(m*n), []uint32{uint32(m), uint32(k), uint32(n), mode})
}
func (op *VkTrainingF32) SiLUDualForwardStage(ctx context.Context, dst, x, xt *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.siluDualForward == nil || dst == nil || x == nil || xt == nil || mode > 1 || dst.rank != 1 || x.rank != 1 || xt.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("dual SiLU forward rank/mode/operator")
	}
	n := x.shape[0]
	if n < 1 || n > 1<<26 || xt.shape[0] != n || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("dual SiLU forward shape")
	}
	return trainingStageLocked(op.siluDualForward, []*VkTensorF32{x, xt, dst}, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(n), mode})
}
func (op *VkTrainingF32) ModDualForwardStage(ctx context.Context, dst, a, at, b, bt *VkTensorF32, kind, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.modDualForward == nil || a == nil || a.rank != 1 || mode > 1 {
		return VkF32Stage{}, fmt.Errorf("dual modulation forward mode/operator")
	}
	n := a.shape[0]
	if err := train.DualModShape(n, kind, mode); err != nil {
		return VkF32Stage{}, err
	}
	bn := n
	if kind == 0 {
		bn *= 2
	}
	tensors := []*VkTensorF32{a, at, b, bt, dst}
	lengths := []int{n, n, bn, bn, n}
	sizes := make([]uint64, 5)
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != lengths[i] {
			return VkF32Stage{}, fmt.Errorf("dual modulation forward shape")
		}
		sizes[i] = uint64(lengths[i]) * 4
	}
	return trainingStageLocked(op.modDualForward, tensors, sizes, []int{4}, uint32(n), []uint32{uint32(n), kind, mode})
}
