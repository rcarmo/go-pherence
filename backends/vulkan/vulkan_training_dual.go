package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *VkTrainingF32) ModDualBackwardStage(ctx context.Context, dst, a, at, b, bt, gv, gt *VkTensorF32, kind, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.modDual == nil || a == nil || a.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid dual modulation operator")
	}
	n := a.shape[0]
	if err := train.DualModShape(n, kind, mode); err != nil {
		return VkF32Stage{}, err
	}
	bn, count := n, n
	if kind == 0 {
		bn = 2 * n
		if mode >= 2 {
			count = 2 * n
		}
	}
	tensors := []*VkTensorF32{a, at, b, bt, gv, gt, dst}
	lengths := []int{n, n, bn, bn, n, n, count}
	sizes := make([]uint64, 7)
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != lengths[i] {
			return VkF32Stage{}, fmt.Errorf("dual modulation shape mismatch")
		}
		sizes[i] = uint64(lengths[i]) * 4
	}
	return trainingStageLocked(op.modDual, tensors, sizes, []int{6}, uint32(count), []uint32{uint32(n), kind, mode})
}

func (op *VkTrainingF32) LinearDualBackwardStage(ctx context.Context, dst, x, xt, w, gv, gt *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.linearDual == nil || dst == nil || x == nil || xt == nil || w == nil || gv == nil || gt == nil || x.rank != 2 || xt.rank != 2 || w.rank != 2 || gv.rank != 2 || gt.rank != 2 || mode > 3 {
		return VkF32Stage{}, fmt.Errorf("invalid dual linear rank/operator/mode")
	}
	m, k, n := x.shape[0], x.shape[1], w.shape[0]
	if err := train.LinearShape(m, k, n); err != nil {
		return VkF32Stage{}, err
	}
	if xt.shape[0] != m || xt.shape[1] != k || w.shape[1] != k || gv.shape[0] != m || gv.shape[1] != n || gt.shape[0] != m || gt.shape[1] != n {
		return VkF32Stage{}, fmt.Errorf("dual linear input mismatch")
	}
	count := n
	if mode < 2 {
		count = m * k
		if dst.rank != 2 || dst.shape[0] != m || dst.shape[1] != k {
			return VkF32Stage{}, fmt.Errorf("dual linear dInput shape")
		}
	} else if mode == 2 {
		count = n * k
		if dst.rank != 2 || dst.shape[0] != n || dst.shape[1] != k {
			return VkF32Stage{}, fmt.Errorf("dual linear dWeight shape")
		}
	} else if dst.rank != 1 || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("dual linear dBias shape")
	}
	return trainingStageLocked(op.linearDual, []*VkTensorF32{x, xt, w, gv, gt, dst}, []uint64{uint64(m*k) * 4, uint64(m*k) * 4, uint64(n*k) * 4, uint64(m*n) * 4, uint64(m*n) * 4, uint64(count) * 4}, []int{5}, uint32(count), []uint32{uint32(m), uint32(k), uint32(n), mode})
}
func (op *VkTrainingF32) NormDualBackwardStage(ctx context.Context, dst, x, xt, w, gv, gt, stats *VkTensorF32, kind, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.normDual == nil || dst == nil || x == nil || xt == nil || w == nil || gv == nil || gt == nil || stats == nil || x.rank != 2 || xt.rank != 2 || gv.rank != 2 || gt.rank != 2 || w.rank != 1 || stats.rank != 2 {
		return VkF32Stage{}, fmt.Errorf("invalid dual normalization operator/rank")
	}
	m, n := x.shape[0], x.shape[1]
	if err := train.DualNormShape(m, n, kind, mode); err != nil {
		return VkF32Stage{}, err
	}
	if xt.shape[0] != m || xt.shape[1] != n || gv.shape[0] != m || gv.shape[1] != n || gt.shape[0] != m || gt.shape[1] != n || w.shape[0] != n || stats.shape[0] != m || stats.shape[1] != 5 {
		return VkF32Stage{}, fmt.Errorf("dual normalization input shape mismatch")
	}
	elements, count := n, n
	if mode < 2 {
		elements = m * n
		count = m
		if dst.rank != 2 || dst.shape[0] != m || dst.shape[1] != n {
			return VkF32Stage{}, fmt.Errorf("dual normalization dInput shape")
		}
	} else if dst.rank != 1 || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("dual normalization parameter gradient shape")
	}
	return trainingStageLocked(op.normDual, []*VkTensorF32{x, xt, w, gv, gt, stats, dst}, []uint64{uint64(m*n) * 4, uint64(m*n) * 4, uint64(n) * 4, uint64(m*n) * 4, uint64(m*n) * 4, uint64(m*5) * 4, uint64(elements) * 4}, []int{6}, uint32(count), []uint32{uint32(m), uint32(n), kind, mode})
}
