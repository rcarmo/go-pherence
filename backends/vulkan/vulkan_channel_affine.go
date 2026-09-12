package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkChannelAffineReLUF32 applies Out[c,s] = fma(X[c,s],Scale[c],Shift[c])
// to contiguous channel-major F32 tensors. Optional ReLU is exact max(value,0)
// for finite inputs. This covers inference BatchNorm coefficients prepared by a
// model owner; it does not derive coefficients, scan data, or change defaults.
// Shapes are X/Out[channels,spatial...] rank2..8 and Scale/Shift[channels].
type VkChannelAffineReLUF32 struct{ kernel *VkComputeKernel }

func NewVkChannelAffineReLUF32(ctx context.Context) (*VkChannelAffineReLUF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_channel_affine_relu_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	return &VkChannelAffineReLUF32{kernel: k}, nil
}
func (op *VkChannelAffineReLUF32) Close() error {
	if op == nil || op.kernel == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage validates exact shape/storage and alias rules before returning a copied
// generic plan stage. Out may exactly alias X. Scale/Shift may alias each other,
// but neither may overlap Out; partial X/Out overlap is rejected.
func (op *VkChannelAffineReLUF32) Stage(ctx context.Context, out, x, scale, shift *VkTensorF32, relu bool) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, scale, shift, relu)
}
func (op *VkChannelAffineReLUF32) stageLocked(out, x, scale, shift *VkTensorF32, relu bool) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan ChannelAffineReLUF32: %s", s)
	}
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{x, scale, shift, out}
	bindings := make([]vkBufferBinding, len(tensors))
	for i, tensor := range tensors {
		binding, err := tensor.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = binding
	}
	if x.rank < 2 || x.rank > 8 || out.rank != x.rank || scale.rank != 1 || shift.rank != 1 {
		return fail("rank mismatch")
	}
	for i := 0; i < x.rank; i++ {
		if out.shape[i] != x.shape[i] {
			return fail("shape mismatch")
		}
	}
	channels := x.shape[0]
	if channels < 1 || channels > 2048 || scale.shape[0] != channels || shift.shape[0] != channels {
		return fail("channel mismatch")
	}
	bytes, err := vkF32ShapeBytes(x.shape[:x.rank])
	if err != nil {
		return VkF32Stage{}, err
	}
	if bytes != bindings[0].size || bytes != bindings[3].size || bindings[1].size != uint64(channels)*4 || bindings[2].size != uint64(channels)*4 {
		return fail("shape/storage mismatch")
	}
	count := bytes / 4
	if count == 0 || count > uint64(^uint32(0)) || count%uint64(channels) != 0 {
		return fail("element count overflow")
	}
	spatial := count / uint64(channels)
	if spatial == 0 || spatial > uint64(^uint32(0)) {
		return fail("spatial count overflow")
	}
	if bindings[0] != bindings[3] && vkBindingsOverlap(bindings[0], bindings[3]) {
		return fail("partial output overlap")
	}
	if vkBindingsOverlap(bindings[1], bindings[3]) || vkBindingsOverlap(bindings[2], bindings[3]) {
		return fail("output overlaps coefficients")
	}
	groups := uint32((count + 255) / 256)
	flag := uint32(0)
	if relu {
		flag = 1
	}
	push := []uint32{uint32(channels), uint32(spatial), flag}
	if err := op.kernel.validateBindingsLocked(groups, 1, 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{groups, 1, 1}, Tensors: tensors, PushWords: push}, nil
}
func (op *VkChannelAffineReLUF32) Forward(ctx context.Context, out, x, scale, shift *VkTensorF32, relu bool) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x, scale, shift, relu)
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
