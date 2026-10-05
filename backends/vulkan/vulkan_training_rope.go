package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// RoPEStage reads X[rows,heads,dim] and table[rows,dim/2,2], shared across
// heads, and overwrites a disjoint output. mode1 is the input-adjoint R^T.
// No shader trigonometry, native initialisation or implicit fallback occurs.
func (op *VkTrainingF32) RoPEStage(ctx context.Context, dst, x, table *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.rope == nil || x == nil || dst == nil || table == nil || x.rank != 3 || dst.rank != 3 || table.rank != 3 {
		return VkF32Stage{}, fmt.Errorf("invalid training RoPE operator/rank")
	}
	rows, heads, dim := x.shape[0], x.shape[1], x.shape[2]
	if err := train.RoPEShape(rows, heads, dim, mode); err != nil {
		return VkF32Stage{}, err
	}
	if dst.shape[0] != rows || dst.shape[1] != heads || dst.shape[2] != dim || table.shape[0] != rows || table.shape[1] != dim/2 || table.shape[2] != 2 {
		return VkF32Stage{}, fmt.Errorf("training RoPE tensor shape")
	}
	count := rows * heads * dim
	return trainingStageLocked(op.rope, []*VkTensorF32{x, table, dst}, []uint64{uint64(count) * 4, uint64(rows*dim) * 4, uint64(count) * 4}, []int{2}, uint32(count/2), []uint32{uint32(rows), uint32(heads), uint32(dim), mode})
}
