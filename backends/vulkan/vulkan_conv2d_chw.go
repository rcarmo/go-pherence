package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkConv2DCHWF32 applies a bias-free 2-D convolution to contiguous channel-major
// X[inChannels,frequency,frames] and W[outChannels,inChannels,kernel,kernel].
// Out is [outChannels,outFrequency,outFrames]. The checked Community-1 envelope
// admits kernel3/pad1 or kernel1/pad0, stride1/2, no groups or dilation.
// Caller-prepared BatchNorm, activation and residual addition are separate ops.
type VkConv2DCHWF32 struct{ kernel *VkComputeKernel }

func NewVkConv2DCHWF32(ctx context.Context) (*VkConv2DCHWF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_conv2d_chw_f32, 3, 36)
	if err != nil {
		return nil, err
	}
	return &VkConv2DCHWF32{kernel: k}, nil
}

func (op *VkConv2DCHWF32) Close() error {
	if op == nil || op.kernel == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage validates complete shape, storage and alias invariants before returning
// a copied generic stage. Output must be disjoint from input and weights;
// read-only input and weights may share storage. No host content scan is made.
func (op *VkConv2DCHWF32) Stage(ctx context.Context, out, x, weight *VkTensorF32, kernel, stride, padding int) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, weight, kernel, stride, padding)
}

func (op *VkConv2DCHWF32) stageLocked(out, x, weight *VkTensorF32, kernel, stride, padding int) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan Conv2DCHWF32: %s", s)
	}
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	if (kernel != 1 && kernel != 3) || (stride != 1 && stride != 2) || padding != kernel/2 {
		return fail("unsupported kernel/stride/padding")
	}
	tensors := []*VkTensorF32{x, weight, out}
	bindings := make([]vkBufferBinding, len(tensors))
	for i, tensor := range tensors {
		binding, err := tensor.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = binding
	}
	if x.rank != 3 || weight.rank != 4 || out.rank != 3 {
		return fail("rank mismatch")
	}
	inChannels, inFrequency, inFrames := x.shape[0], x.shape[1], x.shape[2]
	outChannels := weight.shape[0]
	if inChannels < 1 || inChannels > 256 || outChannels < 1 || outChannels > 256 || inFrequency < 1 || inFrequency > 80 || inFrames < 1 || inFrames > 4096 {
		return fail("dimension envelope")
	}
	outFrequency := (inFrequency + stride - 1) / stride
	outFrames := (inFrames + stride - 1) / stride
	if weight.shape[1] != inChannels || weight.shape[2] != kernel || weight.shape[3] != kernel || out.shape[0] != outChannels || out.shape[1] != outFrequency || out.shape[2] != outFrames {
		return fail("convolution shape mismatch")
	}
	inElements := uint64(inChannels) * uint64(inFrequency) * uint64(inFrames)
	weightElements := uint64(outChannels) * uint64(inChannels) * uint64(kernel) * uint64(kernel)
	outSpatial := uint64(outFrequency) * uint64(outFrames)
	outElements := uint64(outChannels) * outSpatial
	if inElements > 4*1024*1024 || outElements > 4*1024*1024 || outSpatial > uint64(^uint32(0)) {
		return fail("element envelope")
	}
	sizes := []uint64{inElements * 4, weightElements * 4, outElements * 4}
	for i, binding := range bindings {
		if binding.size != sizes[i] {
			return fail("shape/storage mismatch")
		}
		if i < 2 && vkBindingsOverlap(binding, bindings[2]) {
			return fail("output overlaps input")
		}
	}
	groups := [3]uint32{uint32((outChannels + 31) / 32), uint32((outSpatial + 31) / 32), 1}
	push := []uint32{
		uint32(inChannels), uint32(inFrequency), uint32(inFrames),
		uint32(outChannels), uint32(outFrequency), uint32(outFrames),
		uint32(kernel), uint32(stride), uint32(padding),
	}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: groups, Tensors: tensors, PushWords: push}, nil
}

func (op *VkConv2DCHWF32) Forward(ctx context.Context, out, x, weight *VkTensorF32, kernel, stride, padding int) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x, weight, kernel, stride, padding)
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
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], stage.Groups[1], 1, bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
