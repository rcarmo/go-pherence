package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkLSTMSequenceF32 evaluates one direction of an unprojected PyTorch-IFGO LSTM.
// Input[frames,inputDim], weights[4*hidden,inputDim]/[4*hidden,hidden], biases
// [4*hidden], mutable hidden/cell[hidden], output[frames,outputWidth]. OutputOffset
// selects a disjoint direction slice. One workgroup owns the sequential recurrence.
type VkLSTMSequenceF32 struct{ kernel *VkComputeKernel }

func NewVkLSTMSequenceF32(ctx context.Context) (*VkLSTMSequenceF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	kernel, err := vkKernelCreateLocked(spirv_lstm_sequence_f32, 8, 24)
	if err != nil {
		return nil, err
	}
	return &VkLSTMSequenceF32{kernel: kernel}, nil
}

func (op *VkLSTMSequenceF32) Close() error {
	if op == nil || op.kernel == nil {
		return nil
	}
	return op.kernel.Close()
}

func (op *VkLSTMSequenceF32) Stage(ctx context.Context, output, input, weightIH, weightHH, biasIH, biasHH, hidden, cell *VkTensorF32, outputOffset int, reverse bool) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(output, input, weightIH, weightHH, biasIH, biasHH, hidden, cell, outputOffset, reverse)
}

func (op *VkLSTMSequenceF32) stageLocked(output, input, weightIH, weightHH, biasIH, biasHH, hidden, cell *VkTensorF32, outputOffset int, reverse bool) (VkF32Stage, error) {
	fail := func(reason string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan LSTMSequenceF32: %s", reason)
	}
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{input, weightIH, weightHH, biasIH, biasHH, hidden, cell, output}
	bindings := make([]vkBufferBinding, len(tensors))
	for i, tensor := range tensors {
		binding, err := tensor.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = binding
	}
	if input.rank != 2 || weightIH.rank != 2 || weightHH.rank != 2 || biasIH.rank != 1 || biasHH.rank != 1 || hidden.rank != 1 || cell.rank != 1 || output.rank != 2 {
		return fail("rank mismatch")
	}
	frames, inputDim, hiddenSize := input.shape[0], input.shape[1], hidden.shape[0]
	outputWidth := output.shape[1]
	if frames < 1 || frames > 4096 || inputDim < 1 || inputDim > 512 || hiddenSize < 1 || hiddenSize > 256 || outputOffset < 0 || outputWidth < hiddenSize || outputOffset > outputWidth-hiddenSize {
		return fail("dimension envelope")
	}
	gates := 4 * hiddenSize
	if weightIH.shape[0] != gates || weightIH.shape[1] != inputDim || weightHH.shape[0] != gates || weightHH.shape[1] != hiddenSize || biasIH.shape[0] != gates || biasHH.shape[0] != gates || cell.shape[0] != hiddenSize || output.shape[0] != frames {
		return fail("LSTM sequence shape mismatch")
	}
	sizes := []uint64{
		uint64(frames) * uint64(inputDim) * 4,
		uint64(gates) * uint64(inputDim) * 4,
		uint64(gates) * uint64(hiddenSize) * 4,
		uint64(gates) * 4,
		uint64(gates) * 4,
		uint64(hiddenSize) * 4,
		uint64(hiddenSize) * 4,
		uint64(frames) * uint64(outputWidth) * 4,
	}
	for i, binding := range bindings {
		if binding.size != sizes[i] {
			return fail("shape/storage mismatch")
		}
		if i < 7 && vkBindingsOverlap(binding, bindings[7]) {
			return fail("output overlaps input/state")
		}
	}
	for i := 0; i < 7; i++ {
		if i == 5 || i == 6 {
			continue
		}
		if vkBindingsOverlap(bindings[i], bindings[5]) || vkBindingsOverlap(bindings[i], bindings[6]) {
			return fail("mutable state overlaps input/weights")
		}
	}
	if vkBindingsOverlap(bindings[5], bindings[6]) {
		return fail("hidden/cell overlap")
	}
	flag := uint32(0)
	if reverse {
		flag = 1
	}
	push := []uint32{uint32(frames), uint32(inputDim), uint32(hiddenSize), uint32(outputWidth), uint32(outputOffset), flag}
	if err := op.kernel.validateBindingsLocked(1, 1, 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{1, 1, 1}, Tensors: tensors, PushWords: push}, nil
}

func (op *VkLSTMSequenceF32) Forward(ctx context.Context, output, input, weightIH, weightHH, biasIH, biasHH, hidden, cell *VkTensorF32, outputOffset int, reverse bool) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(output, input, weightIH, weightHH, biasIH, biasHH, hidden, cell, outputOffset, reverse)
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
	err = op.kernel.dispatchBindingsLocked(ctx, 1, 1, 1, bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
