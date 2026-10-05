package vulkan

import (
	"context"
	"errors"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type VulkanResidentDriver struct {
	Operator *VkTrainingF32
	retained []*VkF32Plan
	terminal error
}
type vkResidentBuffer struct {
	arena      *VkTensorArena
	tensor     *VkTensorF32
	borrowed   bool
	sourceTape *VkTrainingTape
}

func (b *vkResidentBuffer) Upload(ctx context.Context, x []float32) error {
	if b.borrowed {
		return fmt.Errorf("borrowed tape output is read-only")
	}
	return b.tensor.Upload(ctx, x)
}
func (b *vkResidentBuffer) Download(ctx context.Context, x []float32) error {
	if b.sourceTape != nil && (b.sourceTape.closed || b.sourceTape.terminal != nil) {
		return fmt.Errorf("borrowed tape output expired")
	}
	return b.tensor.Download(ctx, x)
}
func (b *vkResidentBuffer) Close() error {
	if b.borrowed {
		return fmt.Errorf("borrowed tape output cannot be closed")
	}
	return b.arena.Close()
}
func (d *VulkanResidentDriver) NewBuffer(ctx context.Context, n int) (train.ResidentBuffer, error) {
	if d == nil || d.Operator == nil || d.terminal != nil || d.Operator.adam == nil || d.Operator.vector == nil || d.Operator.normStats == nil {
		return nil, fmt.Errorf("explicit live Vulkan resident kernels required")
	}
	if _, err := train.NormStatsShape(n); err != nil {
		return nil, err
	}
	a, err := NewVkTensorArena(ctx, n*4)
	if a == nil {
		return nil, err
	}
	owner := &vkResidentBuffer{arena: a}
	if err != nil {
		return owner, err
	}
	owner.tensor, err = a.AllocF32(ctx, n)
	return owner, err
}
func vkResidentBuffers(values ...train.ResidentBuffer) ([]*VkTensorF32, error) {
	out := make([]*VkTensorF32, len(values))
	for i, x := range values {
		b, ok := x.(*vkResidentBuffer)
		if !ok || b == nil || b.tensor == nil || (b.sourceTape != nil && (b.sourceTape.closed || b.sourceTape.terminal != nil)) {
			return nil, fmt.Errorf("foreign/nil Vulkan resident buffer")
		}
		out[i] = b.tensor
	}
	return out, nil
}
func (d *VulkanResidentDriver) run(ctx context.Context, stage VkF32Stage) error {
	if d == nil || d.terminal != nil {
		return fmt.Errorf("nil/terminal Vulkan resident driver")
	}
	p, err := NewVkF32Plan(ctx, []VkF32Stage{stage})
	if err != nil {
		return err
	}
	if err = p.Run(ctx); err != nil {
		d.retained = append(d.retained, p)
		d.terminal = err
		return err
	}
	if err = p.Close(); err != nil {
		d.retained = append(d.retained, p)
		d.terminal = err
		return err
	}
	return nil
}
func (d *VulkanResidentDriver) Accumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/terminal Vulkan resident driver")
	}
	v, err := vkResidentBuffers(dst, a, b)
	if err != nil {
		return err
	}
	for i, x := range v {
		if x.Elements() != n {
			return fmt.Errorf("resident accumulation extent")
		}
		v[i], err = x.ReshapeF32(ctx, n)
		if err != nil {
			return err
		}
	}
	s, err := d.Operator.VectorStage(ctx, v[0], v[1], v[2], 0, 0, scale)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}
func (d *VulkanResidentDriver) NormStats(ctx context.Context, dst, x train.ResidentBuffer, n int, nonnegative bool) error {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/terminal Vulkan resident driver")
	}
	v, err := vkResidentBuffers(dst, x)
	if err != nil {
		return err
	}
	count, e := train.NormStatsShape(n)
	if e != nil {
		return e
	}
	if v[1].rank != 1 || v[1].shape[0] != n || v[0].rank != 1 || v[0].shape[0] != count {
		return fmt.Errorf("resident norm extent")
	}
	s, err := d.Operator.NormStatsStage(ctx, v[0], v[1], nonnegative)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}
func (d *VulkanResidentDriver) ValidateUpdate(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, step train.AdamWStep) error {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/terminal Vulkan resident driver")
	}
	b, err := vkResidentBuffers(p, g, m, v, ema)
	if err != nil {
		return err
	}
	for _, x := range b {
		if x.rank != 1 || x.shape[0] != n {
			return fmt.Errorf("Vulkan resident update extent")
		}
	}
	_, err = d.Operator.AdamWStage(ctx, b[0], b[1], b[2], b[3], b[4], step)
	return err
}
func (d *VulkanResidentDriver) Update(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, step train.AdamWStep) error {
	if err := d.ValidateUpdate(ctx, p, g, m, v, ema, n, step); err != nil {
		return err
	}
	b, _ := vkResidentBuffers(p, g, m, v, ema)
	s, err := d.Operator.AdamWStage(ctx, b[0], b[1], b[2], b[3], b[4], step)
	if err != nil {
		return err
	}
	return d.run(ctx, s)
}

// Close owns no tensor storage; failed plans keep their kernels/tensors reachable.
func (d *VulkanResidentDriver) Close() error {
	if d == nil {
		return nil
	}
	if d.terminal != nil {
		return fmt.Errorf("Vulkan resident plans retained: %w", d.terminal)
	}
	var err error
	for _, p := range d.retained {
		err = errors.Join(err, p.Close())
	}
	return err
}
func (op *VkTrainingF32) NormStatsStage(ctx context.Context, dst, x *VkTensorF32, nonnegative bool) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.normStats == nil || x == nil || dst == nil || x.rank != 1 || dst.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid norm statistics rank/operator")
	}
	n := x.shape[0]
	count, err := train.NormStatsShape(n)
	if err != nil {
		return VkF32Stage{}, err
	}
	if dst.shape[0] != count {
		return VkF32Stage{}, fmt.Errorf("norm statistics output shape")
	}
	positive := uint32(0)
	if nonnegative {
		positive = 1
	}
	return trainingStageLocked(op.normStats, []*VkTensorF32{x, dst}, []uint64{uint64(n) * 4, uint64(count) * 4}, []int{1}, uint32(count/3), []uint32{uint32(n), positive})
}
