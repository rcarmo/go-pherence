package vulkan

import (
	"context"
)

// LinearDualBackwardPlan overwrites primal/tangent input, weight and optional
// bias adjoints. All write ranges are independent. No accumulation or tangent
// bias is inserted. Pipeline/arena lifetime remains with the caller.
func (op *VkTrainingF32) LinearDualBackwardPlan(ctx context.Context, dx, dt, dw, db, x, xt, w, gv, gt *VkTensorF32) (*VkF32Plan, error) {
	outputs := []*VkTensorF32{dx, dt, dw}
	if db != nil {
		outputs = append(outputs, db)
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	err := trainingDistinctLocked(outputs, []*VkTensorF32{x, xt, w, gv, gt})
	vkRelease()
	if err != nil {
		return nil, err
	}
	var stages []VkF32Stage
	for mode, dst := range outputs {
		s, err := op.LinearDualBackwardStage(ctx, dst, x, xt, w, gv, gt, uint32(mode))
		if err != nil {
			return nil, err
		}
		stages = append(stages, s)
	}
	return NewVkF32Plan(ctx, stages)
}

// NormDualBackwardPlan writes input adjoints and affine gradients. kind1 is
// sample-variance norm and has no bias (db must be nil). Stats come from forward.
func (op *VkTrainingF32) NormDualBackwardPlan(ctx context.Context, dx, dt, dw, db, x, xt, w, gv, gt, stats *VkTensorF32, kind uint32) (*VkF32Plan, error) {
	outputs := []*VkTensorF32{dx, dt, dw}
	if db != nil {
		outputs = append(outputs, db)
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	err := trainingDistinctLocked(outputs, []*VkTensorF32{x, xt, w, gv, gt, stats})
	vkRelease()
	if err != nil {
		return nil, err
	}
	var stages []VkF32Stage
	for mode, dst := range outputs {
		s, err := op.NormDualBackwardStage(ctx, dst, x, xt, w, gv, gt, stats, kind, uint32(mode))
		if err != nil {
			return nil, err
		}
		stages = append(stages, s)
	}
	return NewVkF32Plan(ctx, stages)
}
