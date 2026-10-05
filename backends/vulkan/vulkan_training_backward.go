package vulkan

import (
	"context"
	"fmt"
	"math"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// SiLUDualBackwardStage evaluates reverse-over-forward SiLU input adjoints
// from primal/tangent inputs and independent output adjoints. It retains the
// second derivative needed by the minimal-LSD time-JVP loss, not finite differences.
func (op *VkTrainingF32) SiLUDualBackwardStage(ctx context.Context, dx, dt, x, xt, gv, gt *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.siluDual == nil || x == nil || x.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid dual SiLU operator")
	}
	n := x.shape[0]
	if n < 1 || n > 1<<26 {
		return VkF32Stage{}, fmt.Errorf("dual SiLU vector ceiling")
	}
	tensors := []*VkTensorF32{x, xt, gv, gt, dx, dt}
	sizes := make([]uint64, 6)
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != n {
			return VkF32Stage{}, fmt.Errorf("dual SiLU shape mismatch")
		}
		sizes[i] = uint64(n) * 4
	}
	return trainingStageLocked(op.siluDual, tensors, sizes, []int{4, 5}, uint32(n), []uint32{uint32(n)})
}

// LayerNormBackwardStage consumes cached pre-affine Xhat, inverse stddev and
// affine weight. Each dispatch overwrites one derivative; no atomic addition.
func (op *VkTrainingF32) LayerNormBackwardStage(ctx context.Context, dst, xhat, inv, weight, dy *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.norm == nil || dst == nil || xhat == nil || inv == nil || weight == nil || dy == nil || xhat.rank != 2 || inv.rank != 1 || weight.rank != 1 || dy.rank != 2 {
		return VkF32Stage{}, fmt.Errorf("invalid layernorm backward rank/operator")
	}
	rows, width := xhat.shape[0], xhat.shape[1]
	if err := train.LayerNormShape(rows, width, mode); err != nil {
		return VkF32Stage{}, err
	}
	if inv.shape[0] != rows || weight.shape[0] != width || dy.shape[0] != rows || dy.shape[1] != width {
		return VkF32Stage{}, fmt.Errorf("layernorm backward input mismatch")
	}
	elements, count := width, width
	if mode == 0 {
		elements = rows * width
		count = rows
		if dst.rank != 2 || dst.shape[0] != rows || dst.shape[1] != width {
			return VkF32Stage{}, fmt.Errorf("layernorm dX shape")
		}
	} else if dst.rank != 1 || dst.shape[0] != width {
		return VkF32Stage{}, fmt.Errorf("layernorm parameter gradient shape")
	}
	return trainingStageLocked(op.norm, []*VkTensorF32{xhat, inv, weight, dy, dst}, []uint64{uint64(rows) * uint64(width) * 4, uint64(rows) * 4, uint64(width) * 4, uint64(rows) * uint64(width) * 4, uint64(elements) * 4}, []int{4}, uint32(count), []uint32{uint32(rows), uint32(width), mode})
}

// AttentionBackwardStage is one causal backward primitive, not a complete
// transformer. Mode0 produces dense dP; mode1 consumes it to produce scaled
// dScore; mode2/3 consume dScore for dQ/dK; mode4 produces dV. Only mode0/1
// write dense [H,S,S]; all other outputs are [S,H,D]. No padding keys.
// Aux must still be a distinct valid [H,S,S] tensor when unused (mode0/4).
func (op *VkTrainingF32) AttentionBackwardStage(ctx context.Context, dst, q, k, v, prob, dy, aux *VkTensorF32, contextRows int, mode uint32, scale float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.attention == nil || dst == nil || q == nil || q.rank != 3 {
		return VkF32Stage{}, fmt.Errorf("invalid attention backward rank/operator")
	}
	rows, heads, dim := q.shape[0], q.shape[1], q.shape[2]
	if err := train.AttentionShape(rows, heads, dim, contextRows, mode, scale); err != nil {
		return VkF32Stage{}, err
	}
	for _, t := range []*VkTensorF32{k, v, dy} {
		if t == nil || t.rank != 3 || t.shape[0] != rows || t.shape[1] != heads || t.shape[2] != dim {
			return VkF32Stage{}, fmt.Errorf("attention QKV/dO shape mismatch")
		}
	}
	for _, t := range []*VkTensorF32{prob, aux} {
		if t == nil || t.rank != 3 || t.shape[0] != heads || t.shape[1] != rows || t.shape[2] != rows {
			return VkF32Stage{}, fmt.Errorf("attention probability/aux shape mismatch")
		}
	}
	dense := heads * rows * rows
	elements, count := rows*heads*dim, rows*heads*dim
	if mode < 2 {
		elements = dense
		count = dense
		if mode == 1 {
			count = heads * rows
		}
		if dst.rank != 3 || dst.shape[0] != heads || dst.shape[1] != rows || dst.shape[2] != rows {
			return VkF32Stage{}, fmt.Errorf("attention dense derivative shape")
		}
	} else if dst.rank != 3 || dst.shape[0] != rows || dst.shape[1] != heads || dst.shape[2] != dim {
		return VkF32Stage{}, fmt.Errorf("attention QKV derivative shape")
	}
	tensorBytes, denseBytes := uint64(rows)*uint64(heads)*uint64(dim)*4, uint64(dense)*4
	return trainingStageLocked(op.attention, []*VkTensorF32{q, k, v, prob, dy, aux, dst}, []uint64{tensorBytes, tensorBytes, tensorBytes, denseBytes, tensorBytes, denseBytes, uint64(elements) * 4}, []int{6}, uint32(count), []uint32{uint32(rows), uint32(heads), uint32(dim), uint32(contextRows), mode, math.Float32bits(scale)})
}
