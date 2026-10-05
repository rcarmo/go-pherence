package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// AttentionForwardStage overwrites exact-key causal scores, softmax or weighted
// values. Q/K/V are [S,H,D], aux is [H,S,S] and never aliases writable output.
// mode0 aux is unused scratch; mode1 consumes scores; mode2 probabilities.
func (op *VkTrainingF32) AttentionForwardStage(ctx context.Context, dst, q, k, v, aux *VkTensorF32, contextRows int, mode uint32, scale float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.attentionForward == nil || q == nil || q.rank != 3 || dst == nil {
		return VkF32Stage{}, fmt.Errorf("invalid attention forward rank/operator")
	}
	rows, heads, dim := q.shape[0], q.shape[1], q.shape[2]
	if err := train.AttentionForwardShape(rows, heads, dim, contextRows, mode, scale); err != nil {
		return VkF32Stage{}, err
	}
	for _, t := range []*VkTensorF32{k, v} {
		if t == nil || t.rank != 3 || t.shape[0] != rows || t.shape[1] != heads || t.shape[2] != dim {
			return VkF32Stage{}, fmt.Errorf("attention forward QKV shape")
		}
	}
	if aux == nil || aux.rank != 3 || aux.shape[0] != heads || aux.shape[1] != rows || aux.shape[2] != rows {
		return VkF32Stage{}, fmt.Errorf("attention forward aux shape")
	}
	tensor, dense := rows*heads*dim, heads*rows*rows
	elements, count := dense, dense
	expected := []int{heads, rows, rows}
	if mode == 1 {
		count = heads * rows
	}
	if mode == 2 {
		elements, count = tensor, tensor
		expected = []int{rows, heads, dim}
	}
	if dst.rank != 3 || dst.shape[0] != expected[0] || dst.shape[1] != expected[1] || dst.shape[2] != expected[2] {
		return VkF32Stage{}, fmt.Errorf("attention forward output shape")
	}
	return trainingStageLocked(op.attentionForward, []*VkTensorF32{q, k, v, aux, dst}, []uint64{uint64(tensor) * 4, uint64(tensor) * 4, uint64(tensor) * 4, uint64(dense) * 4, uint64(elements) * 4}, []int{4}, uint32(count), []uint32{uint32(rows), uint32(heads), uint32(dim), uint32(contextRows), mode, math.Float32bits(scale)})
}
