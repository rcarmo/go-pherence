package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkLSTMCellF32 updates one unprojected LSTM state from caller-precombined
// PyTorch-IFGO gates. Gates is [4,hidden]; CellIn/HiddenOut/CellOut are [hidden].
// Exact CellOut/CellIn alias is supported. HiddenOut and CellOut must be disjoint.
// Projection, bias composition, sequence order and bidirectionality are separate.
type VkLSTMCellF32 struct{ kernel *VkComputeKernel }

func NewVkLSTMCellF32(ctx context.Context) (*VkLSTMCellF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	kernel, err := vkKernelCreateLocked(spirv_lstm_cell_f32, 4, 4)
	if err != nil {
		return nil, err
	}
	return &VkLSTMCellF32{kernel: kernel}, nil
}

func (op *VkLSTMCellF32) Close() error {
	if op == nil || op.kernel == nil {
		return nil
	}
	return op.kernel.Close()
}

func (op *VkLSTMCellF32) Stage(ctx context.Context, hiddenOut, cellOut, gates, cellIn *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(hiddenOut, cellOut, gates, cellIn)
}

func (op *VkLSTMCellF32) stageLocked(hiddenOut, cellOut, gates, cellIn *VkTensorF32) (VkF32Stage, error) {
	fail := func(reason string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan LSTMCellF32: %s", reason)
	}
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{gates, cellIn, hiddenOut, cellOut}
	bindings := make([]vkBufferBinding, len(tensors))
	for i, tensor := range tensors {
		binding, err := tensor.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = binding
	}
	if gates.rank != 2 || cellIn.rank != 1 || hiddenOut.rank != 1 || cellOut.rank != 1 || gates.shape[0] != 4 {
		return fail("rank/gate shape mismatch")
	}
	hidden := gates.shape[1]
	if hidden < 1 || hidden > 256 || cellIn.shape[0] != hidden || hiddenOut.shape[0] != hidden || cellOut.shape[0] != hidden {
		return fail("hidden shape mismatch")
	}
	sizes := []uint64{uint64(4 * hidden * 4), uint64(hidden * 4), uint64(hidden * 4), uint64(hidden * 4)}
	for i, binding := range bindings {
		if binding.size != sizes[i] {
			return fail("shape/storage mismatch")
		}
	}
	if vkBindingsOverlap(bindings[0], bindings[2]) || vkBindingsOverlap(bindings[0], bindings[3]) || vkBindingsOverlap(bindings[2], bindings[3]) {
		return fail("output overlap")
	}
	if bindings[1] != bindings[3] && vkBindingsOverlap(bindings[1], bindings[3]) {
		return fail("partial cell overlap")
	}
	if vkBindingsOverlap(bindings[1], bindings[2]) {
		return fail("hidden output overlaps cell input")
	}
	groups := uint32((hidden + 255) / 256)
	push := []uint32{uint32(hidden)}
	if err := op.kernel.validateBindingsLocked(groups, 1, 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{groups, 1, 1}, Tensors: tensors, PushWords: push}, nil
}

func (op *VkLSTMCellF32) Forward(ctx context.Context, hiddenOut, cellOut, gates, cellIn *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(hiddenOut, cellOut, gates, cellIn)
	if err != nil {
		return err
	}
	bindings := make([]vkBufferBinding, len(stage.Tensors))
	for i, tensor := range stage.Tensors {
		bindings[i], err = tensor.bindingLocked()
		if err != nil {
			return err
		}
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], 1, 1, bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
