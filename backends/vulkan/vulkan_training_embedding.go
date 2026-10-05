package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// Raw callers must admit IDs/index contents; Tape.Validate does so for tapes.
func (op *VkTrainingF32) EmbeddingStage(ctx context.Context, dst, source, ids *VkTensorF32, rows, width, vocabulary int, mode uint32) (VkF32Stage, error) {
	a, n, err := train.EmbeddingShape(rows, width, vocabulary, mode)
	if err != nil {
		return VkF32Stage{}, err
	}
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	indexSize := rows
	if mode == 2 {
		indexSize += vocabulary + 1
	}
	if mode == 3 {
		indexSize = 3*rows + 1
	}
	if op == nil || op.embedding == nil || source == nil || ids == nil || dst == nil || source.rank != 1 || ids.rank != 1 || dst.rank != 1 || source.shape[0] != a || ids.shape[0] != indexSize || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("training embedding owner/shape")
	}
	return trainingStageLocked(op.embedding, []*VkTensorF32{source, ids, dst}, []uint64{uint64(a) * 4, uint64(indexSize) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(rows), uint32(width), uint32(vocabulary), mode})
}
