package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkConvInputLayout is the physical contiguous layout of Conv1D3 input.
type VkConvInputLayout uint32

const (
	VkConvChannelsFirst VkConvInputLayout = iota // X[inCh,inLen]
	VkConvTimeMajor                              // X[inLen,inCh]
)

// VkConv1D3F32 owns tiled kernel3/pad1 convolution, stride1 or2, no groups or
// dilation. W[outCh,inCh,3], B[outCh], output[ceil(inLen/stride),outCh].
// Both input layouts produce time-major output, allowing resident stem chains.
// inLen1..4096,inCh/outCh1..2048. Caller ensures finite representable dot sums;
// no content scan, quantisation, implicit activation or model/default change.
type VkConv1D3F32 struct{ kernel *VkComputeKernel }

func NewVkConv1D3F32(ctx context.Context) (*VkConv1D3F32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_conv1d3_f32, 4, 24)
	if err != nil {
		return nil, err
	}
	return &VkConv1D3F32{kernel: k}, nil
}
func (op *VkConv1D3F32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage rejects all output/input overlap and checks exact shape/byte extents.
// Input/input alias is allowed. Returned generic stage slices are caller-owned.
// Plans validate generic resources only, not these convolution invariants;
// callers who modify a stage assume responsibility for shader shape/alias safety.
func (op *VkConv1D3F32) Stage(ctx context.Context, out, x, w, b *VkTensorF32, stride int, layout VkConvInputLayout) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, w, b, stride, layout)
}
func (op *VkConv1D3F32) stageLocked(out, x, w, b *VkTensorF32, stride int, layout VkConvInputLayout) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan Conv1D3F32: %s", s) }
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	if (stride != 1 && stride != 2) || (layout != VkConvChannelsFirst && layout != VkConvTimeMajor) {
		return fail("invalid stride/layout")
	}
	tensors := []*VkTensorF32{x, w, b, out}
	bindings := make([]vkBufferBinding, 4)
	for i, t := range tensors {
		v, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = v
	}
	if x.rank != 2 || w.rank != 3 || b.rank != 1 || out.rank != 2 {
		return fail("rank mismatch")
	}
	inCh, inLen := x.shape[0], x.shape[1]
	if layout == VkConvTimeMajor {
		inLen, inCh = x.shape[0], x.shape[1]
	}
	outCh := w.shape[0]
	if inLen < 1 || inLen > 4096 || inCh < 1 || inCh > 2048 || outCh < 1 || outCh > 2048 {
		return fail("dimension envelope")
	}
	outLen := (inLen + stride - 1) / stride
	if w.shape[1] != inCh || w.shape[2] != 3 || b.shape[0] != outCh || out.shape[0] != outLen || out.shape[1] != outCh {
		return fail("convolution shape mismatch")
	}
	sizes := [4]uint64{uint64(inLen) * uint64(inCh) * 4, uint64(outCh) * uint64(inCh) * 3 * 4, uint64(outCh) * 4, uint64(outLen) * uint64(outCh) * 4}
	for i, v := range bindings {
		if v.size != sizes[i] {
			return fail("shape/storage mismatch")
		}
		if i < 3 && vkBindingsOverlap(v, bindings[3]) {
			return fail("output overlaps input")
		}
	}
	groups := [3]uint32{uint32((outCh + 15) / 16), uint32((outLen + 15) / 16), 1}
	push := []uint32{uint32(inLen), uint32(inCh), uint32(outCh), uint32(outLen), uint32(stride), uint32(layout)}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: groups, Tensors: tensors, PushWords: push}, nil
}
func (op *VkConv1D3F32) Forward(ctx context.Context, out, x, w, b *VkTensorF32, stride int, layout VkConvInputLayout) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	s, err := op.stageLocked(out, x, w, b, stride, layout)
	if err != nil {
		return err
	}
	bindings := make([]vkBufferBinding, 4)
	for i, t := range s.Tensors {
		bindings[i], err = t.bindingLocked()
		if err != nil {
			return err
		}
	}
	err = op.kernel.dispatchBindingsLocked(ctx, s.Groups[0], s.Groups[1], 1, bindings, unsafePushWords(s.PushWords))
	runtime.KeepAlive(s.PushWords)
	return err
}
