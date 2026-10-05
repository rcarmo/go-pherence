package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type vkResidentTape struct {
	tape *VkTrainingTape
	ran  bool
}

func (d *VulkanResidentDriver) NewBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, t, parameters, maxBytes, false, false)
}
func (d *VulkanResidentDriver) NewReuseBoundTape(ctx context.Context, tape train.Tape, bindings map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, tape, bindings, maxBytes, true, false)
}
func (d *VulkanResidentDriver) NewZeroBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, t, parameters, maxBytes, true, true)
}
func (d *VulkanResidentDriver) newBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64, reuse, shareZero bool) (train.ResidentBoundTape, error) {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return nil, fmt.Errorf("nil/terminal resident driver")
	}
	borrowed := map[int]*VkTensorF32{}
	for id, input := range parameters {
		v, err := vkResidentBuffers(input)
		if err != nil {
			return nil, err
		}
		borrowed[id] = v[0]
	}
	tape, err := newVkTrainingTapeBound(ctx, d.Operator, t, borrowed, maxBytes, reuse, shareZero)
	if tape == nil {
		return nil, err
	}
	return &vkResidentTape{tape: tape}, err
}
func (d *VulkanResidentDriver) ValidateAccumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	if d == nil || d.Operator == nil || d.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/terminal resident driver")
	}
	values, err := vkResidentBuffers(dst, a, b)
	if err != nil {
		return err
	}
	for _, v := range values {
		if v.Elements() != n {
			return fmt.Errorf("resident accumulation extent")
		}
	}
	flat := make([]*VkTensorF32, 3)
	for i, v := range values {
		flat[i], err = v.ReshapeF32(ctx, n)
		if err != nil {
			return err
		}
	}
	_, err = d.Operator.VectorStage(ctx, flat[0], flat[1], flat[2], 0, 0, scale)
	return err
}
func (t *vkResidentTape) RunDevice(ctx context.Context) error {
	if t == nil || t.tape == nil || t.ran {
		return fmt.Errorf("resident tape already executed/nil")
	}
	if err := t.tape.runDevice(ctx); err != nil {
		return err
	}
	t.ran = true
	return nil
}
func (t *vkResidentTape) OutputBuffer(id int) (train.ResidentBuffer, error) {
	if t != nil && t.tape != nil && t.tape.reuse && !train.DeclaredTapeOutput(t.tape.outputs, id) {
		return nil, fmt.Errorf("lifetime tape buffer must be declared output")
	}
	if t == nil || t.tape == nil || !t.ran || t.tape.closed || t.tape.terminal != nil || id < 0 || id >= len(t.tape.tensors) {
		return nil, fmt.Errorf("resident output unavailable")
	}
	tensor := t.tape.tensors[id]
	return &vkResidentBuffer{tensor: tensor, arena: &VkTensorArena{state: tensor.arena}, borrowed: true, sourceTape: t.tape}, nil
}
func (t *vkResidentTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if t == nil || t.tape == nil || !t.ran {
		return nil, fmt.Errorf("resident tape not executed")
	}
	return t.tape.downloadOutputs(ctx, ids)
}
func (t *vkResidentTape) Close() error {
	if t == nil {
		return nil
	}
	return t.tape.Close()
}
