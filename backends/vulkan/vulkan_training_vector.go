package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

func (op *VkTrainingF32) VectorStage(ctx context.Context, dst, a, b *VkTensorF32, offset int, mode uint32, scale float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.vector == nil || dst == nil || a == nil || b == nil || dst.rank != 1 || a.rank != 1 || b.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid training vector operator/rank")
	}
	n, source := dst.shape[0], a.shape[0]
	if err := train.VectorShape(n, source, offset, mode, scale); err != nil {
		return VkF32Stage{}, err
	}
	// B is unused in modes1/2 but remains a real owned binding. For mode0 it
	// must have the exact output shape; no writable input alias is permitted.
	if b.shape[0] < 1 || (mode == 0 && b.shape[0] != n) {
		return VkF32Stage{}, fmt.Errorf("training vector second input shape")
	}
	return trainingStageLocked(op.vector, []*VkTensorF32{a, b, dst}, []uint64{uint64(source) * 4, uint64(b.shape[0]) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(n), uint32(source), uint32(offset), mode, math.Float32bits(scale)})
}
