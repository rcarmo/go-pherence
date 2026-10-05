package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

func (op *VkTrainingF32) NormDualForwardStage(ctx context.Context, dst, x, xt, w, bias, stats *VkTensorF32, kind, mode uint32, epsilon float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.normDualForward == nil || dst == nil || x == nil || xt == nil || w == nil || bias == nil || stats == nil || x.rank != 2 || xt.rank != 2 || w.rank != 1 || bias.rank != 1 || stats.rank != 2 {
		return VkF32Stage{}, fmt.Errorf("invalid dual norm forward operator/rank")
	}
	m, n := x.shape[0], x.shape[1]
	if err := train.DualNormForwardShape(m, n, kind, mode, epsilon); err != nil {
		return VkF32Stage{}, err
	}
	if xt.shape[0] != m || xt.shape[1] != n || w.shape[0] != n || bias.shape[0] != n || stats.shape[0] != m || stats.shape[1] != 5 {
		return VkF32Stage{}, fmt.Errorf("dual norm forward shape mismatch")
	}
	elements, count := m*n, m*n
	readStats := stats
	if mode == 0 {
		elements = m * 5
		count = m
		if dst.rank != 2 || dst.shape[0] != m || dst.shape[1] != 5 {
			return VkF32Stage{}, fmt.Errorf("dual norm stats output shape")
		}
		readStats = stats
	} else if dst.rank != 2 || dst.shape[0] != m || dst.shape[1] != n {
		return VkF32Stage{}, fmt.Errorf("dual norm forward output shape")
	}
	// mode0 still binds distinct scratch stats; it is not read. This keeps
	// descriptor/alias rules uniform without permitting read/write overlap.
	return trainingStageLocked(op.normDualForward, []*VkTensorF32{x, xt, w, bias, readStats, dst}, []uint64{uint64(m*n) * 4, uint64(m*n) * 4, uint64(n) * 4, uint64(n) * 4, uint64(m*5) * 4, uint64(elements) * 4}, []int{5}, uint32(count), []uint32{uint32(m), uint32(n), kind, mode, math.Float32bits(epsilon)})
}

// NormDualForwardPlan writes stats/value/tangent. Unused stats input in the
// first stage is a distinct scratch view, not an output alias.
func (op *VkTrainingF32) NormDualForwardPlan(ctx context.Context, y, yt, stats, scratch, x, xt, w, bias *VkTensorF32, kind uint32, epsilon float32) (*VkF32Plan, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	err := trainingDistinctLocked([]*VkTensorF32{y, yt, stats}, []*VkTensorF32{scratch, x, xt, w, bias})
	vkRelease()
	if err != nil {
		return nil, err
	}
	var stages []VkF32Stage
	for _, a := range []struct {
		dst, stats *VkTensorF32
		mode       uint32
	}{{stats, scratch, 0}, {y, stats, 1}, {yt, stats, 2}} {
		s, err := op.NormDualForwardStage(ctx, a.dst, x, xt, w, bias, a.stats, kind, a.mode, epsilon)
		if err != nil {
			return nil, err
		}
		stages = append(stages, s)
	}
	return NewVkF32Plan(ctx, stages)
}
