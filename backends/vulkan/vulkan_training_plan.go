package vulkan

import (
	"context"
	"fmt"
)

// AttentionBackwardPlan validates all five stages before any submission. dP and
// dScore are scratch; dQ/dK/dV are pre-inverse-RoPE outputs. The existing Vulkan
// plan supplies compute/host dependencies and retains unresolved submissions.
// All five write ranges must be disjoint from each other and every read-only
// input, even though individual stages only bind their own dependencies.
func (op *VkTrainingF32) AttentionBackwardPlan(ctx context.Context, dq, dk, dv, dp, ds, q, k, v, prob, dy *VkTensorF32, contextRows int, scale float32) (*VkF32Plan, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	err := trainingDistinctLocked([]*VkTensorF32{dq, dk, dv, dp, ds}, []*VkTensorF32{q, k, v, prob, dy})
	vkRelease()
	if err != nil {
		return nil, err
	}
	// Stage admission creates owned metadata, no native launch. Plan admission
	// revalidates bindings/owners under its own serialized lane before recording.
	var stages []VkF32Stage
	for _, stage := range []struct {
		dst, aux *VkTensorF32
		mode     uint32
	}{{dp, ds, 0}, {ds, dp, 1}, {dq, ds, 2}, {dk, ds, 3}, {dv, ds, 4}} {
		s, err := op.AttentionBackwardStage(ctx, stage.dst, q, k, v, prob, dy, stage.aux, contextRows, stage.mode, scale)
		if err != nil {
			return nil, err
		}
		stages = append(stages, s)
	}
	return NewVkF32Plan(ctx, stages)
}

// LayerNormBackwardPlan overwrites all input/affine derivatives in one validated
// plan. It consumes pre-affine normalized values and the forward inverse stddev.
func (op *VkTrainingF32) LayerNormBackwardPlan(ctx context.Context, dx, dw, db, xhat, inv, weight, dy *VkTensorF32) (*VkF32Plan, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	err := trainingDistinctLocked([]*VkTensorF32{dx, dw, db}, []*VkTensorF32{xhat, inv, weight, dy})
	vkRelease()
	if err != nil {
		return nil, err
	}
	var stages []VkF32Stage
	for mode, dst := range []*VkTensorF32{dx, dw, db} {
		stage, err := op.LayerNormBackwardStage(ctx, dst, xhat, inv, weight, dy, uint32(mode))
		if err != nil {
			return nil, err
		}
		stages = append(stages, stage)
	}
	return NewVkF32Plan(ctx, stages)
}
func trainingDistinctLocked(outputs, inputs []*VkTensorF32) error {
	bindings := make([]vkBufferBinding, 0, len(outputs)+len(inputs))
	for _, t := range append(append([]*VkTensorF32(nil), outputs...), inputs...) {
		if t == nil {
			return fmt.Errorf("nil composite training tensor")
		}
		b, err := t.bindingLocked()
		if err != nil {
			return err
		}
		bindings = append(bindings, b)
	}
	for i := range outputs {
		for j := range bindings {
			if i != j && vkBindingsOverlap(bindings[i], bindings[j]) {
				return fmt.Errorf("composite training write ranges overlap")
			}
		}
	}
	return nil
}
