package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// GELUStage applies tanh-approximate GELU or its VJP to disjoint F32 vectors.
// As with other raw stages, callers own finite/representable input admission.
func (op *VkTrainingF32) GELUStage(ctx context.Context, dst, x, seed *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.gelu == nil || x == nil || dst == nil || seed == nil || x.rank != 1 || dst.rank != 1 || seed.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid training GELU rank/operator")
	}
	n := x.shape[0]
	if err := train.GELUShape(n, mode); err != nil {
		return VkF32Stage{}, err
	}
	if dst.shape[0] != n || seed.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("training GELU shape")
	}
	return trainingStageLocked(op.gelu, []*VkTensorF32{x, seed, dst}, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(n), mode})
}

// RowStage accepts flat contiguous vectors; row/part geometry is explicit so
// reshape aliases cannot silently change the gather/scatter convention.
func (op *VkTrainingF32) RowStage(ctx context.Context, dst, a, b *VkTensorF32, rows, width, parts, slot int, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.row == nil || a == nil || b == nil || dst == nil || a.rank != 1 || b.rank != 1 || dst.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid training row rank/operator")
	}
	an, bn, n, err := train.RowShape(rows, width, parts, slot, mode)
	if err != nil {
		return VkF32Stage{}, err
	}
	if a.shape[0] != an || b.shape[0] != bn || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("training row shape")
	}
	return trainingStageLocked(op.row, []*VkTensorF32{a, b, dst}, []uint64{uint64(an) * 4, uint64(bn) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(rows), uint32(width), uint32(parts), uint32(slot), mode})
}
