package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// Index content admission is the raw caller's responsibility; all four buffers
// must stay immutable/alive through the synchronous resident plan execution.
func (op *VkTrainingF32) CompactEmbeddingAccumulateStage(ctx context.Context, out, previous, compact, index *VkTensorF32, rows, width, vocabulary int, scale float32) (VkF32Stage, error) {
	dense, slots, indices, err := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return VkF32Stage{}, err
	}
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.embeddingAccumulate == nil || out == nil || previous == nil || compact == nil || index == nil || out.rank != 1 || previous.rank != 1 || compact.rank != 1 || index.rank != 1 || out.shape[0] != dense || previous.shape[0] != dense || compact.shape[0] != slots || index.shape[0] != indices {
		return VkF32Stage{}, fmt.Errorf("compact accumulation owner/shape")
	}
	return trainingStageLocked(op.embeddingAccumulate, []*VkTensorF32{previous, compact, index, out}, []uint64{uint64(dense) * 4, uint64(slots) * 4, uint64(indices) * 4, uint64(dense) * 4}, []int{3}, uint32(dense), []uint32{uint32(rows), uint32(width), uint32(vocabulary), math.Float32bits(scale)})
}

var _ train.ResidentCompactEmbeddingDriver = (*VulkanResidentDriver)(nil)

func (d *VulkanResidentDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if d == nil || d.Operator == nil {
		return fmt.Errorf("nil Vulkan compact accumulation driver")
	}
	if ctx == nil {
		return fmt.Errorf("nil Vulkan compact accumulation context")
	}
	if d.terminal != nil {
		return fmt.Errorf("terminal Vulkan compact accumulation driver")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	dense, slots, indices, err := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	lengths := []int{dense, slots, indices, dense}
	flat := make([]*VkTensorF32, 4)
	for i, b := range []train.ResidentBuffer{previous, compact, index, out} {
		values, e := vkResidentBuffers(b)
		var x *VkTensorF32
		if e == nil {
			x = values[0]
		}
		if e != nil {
			return e
		}
		if x.Elements() != lengths[i] {
			return fmt.Errorf("compact accumulation buffer shape")
		}
		flat[i], e = x.ReshapeF32(ctx, lengths[i])
		if e != nil {
			return e
		}
	}
	_, err = d.Operator.CompactEmbeddingAccumulateStage(ctx, flat[3], flat[0], flat[1], flat[2], rows, width, vocabulary, scale)
	return err
}
func (d *VulkanResidentDriver) CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if err := d.ValidateCompactEmbeddingAccumulate(ctx, out, previous, compact, index, rows, width, vocabulary, scale); err != nil {
		return err
	}
	dense, slots, indices, _ := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	lengths := []int{dense, slots, indices, dense}
	flat := make([]*VkTensorF32, 4)
	for i, b := range []train.ResidentBuffer{previous, compact, index, out} {
		values, err := vkResidentBuffers(b)
		if err != nil {
			return err
		}
		flat[i], err = values[0].ReshapeF32(ctx, lengths[i])
		if err != nil {
			return err
		}
	}
	stage, err := d.Operator.CompactEmbeddingAccumulateStage(ctx, flat[3], flat[0], flat[1], flat[2], rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	return d.run(ctx, stage)
}
