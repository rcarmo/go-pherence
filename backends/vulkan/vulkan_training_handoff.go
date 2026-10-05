package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

var _ train.ResidentHandoffTapeDriver = (*VulkanResidentDriver)(nil)

func (d *VulkanResidentDriver) ValidateHandoffTape(ctx context.Context, t train.Tape, bindings map[int]train.ResidentBuffer, maxBytes int64) error {
	if d == nil || d.Operator == nil || ctx == nil || d.terminal != nil {
		return fmt.Errorf("nil/terminal Vulkan handoff driver/context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	if err := train.ValidateExternalTapeBindings(t, bindings); err != nil {
		return err
	}
	if err := d.Operator.admitTapeKernels(t); err != nil {
		return err
	}
	tensors := map[int]*VkTensorF32{}
	for id, b := range bindings {
		if id < 0 || id >= len(t.Tensors) || (t.Tensors[id].Values == nil && !t.Tensors[id].External) {
			return fmt.Errorf("Vulkan handoff bound constant ID")
		}
		values, err := vkResidentBuffers(b)
		if err != nil {
			return err
		}
		n, _ := t.Tensors[id].Elements()
		if values[0].Elements() != n {
			return fmt.Errorf("Vulkan handoff bound extent")
		}
		tensors[id] = values[0]
	}
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	bound := map[int]vkBufferBinding{}
	for id, tensor := range tensors {
		b, err := tensor.bindingLocked()
		if err != nil {
			return err
		}
		bound[id] = b
	}
	for id, a := range bound {
		for other, b := range bound {
			if other < id && vkBindingsOverlap(a, b) {
				return fmt.Errorf("Vulkan handoff bound overlap")
			}
		}
	}
	return nil
}
