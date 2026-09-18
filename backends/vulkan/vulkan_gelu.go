package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkGELUErfF32 implements the erf GELU form via an explicit erfc approximation,
// not tanh and not bit-exact libm erf. GPU error/rounding and trained-model quality
// remain unqualified; source-model tolerances are documented separately.
// Input data must be finite. No content scan, F16/quantisation or silent fallback.
type VkGELUErfF32 struct{ kernel *VkComputeKernel }

func NewVkGELUErfF32(ctx context.Context) (*VkGELUErfF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_gelu_erf_f32, 2, 4)
	if err != nil {
		return nil, err
	}
	return &VkGELUErfF32{kernel: k}, nil
}
func (op *VkGELUErfF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage/Forward accept identical contiguous F32 shapes of rank1..8. One
// invocation owns one element. Exact in-place alias is supported; partial
// overlap is rejected. The generic returned stage must not be altered to bypass
// these operator checks. Element counts obey uint32/grid/descriptor bounds.
func (op *VkGELUErfF32) Stage(ctx context.Context, out, x *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x)
}
func (op *VkGELUErfF32) stageLocked(out, x *VkTensorF32) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan GELUErfF32: %s", s) }
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	input, err := x.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	output, err := out.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	if x.rank < 1 || x.rank > 8 || out.rank != x.rank {
		return fail("rank mismatch")
	}
	for i := 0; i < x.rank; i++ {
		if out.shape[i] != x.shape[i] {
			return fail("shape mismatch")
		}
	}
	bytes, err := vkF32ShapeBytes(x.shape[:x.rank])
	if err != nil {
		return VkF32Stage{}, err
	}
	if input.size != bytes || output.size != bytes {
		return fail("shape/storage mismatch")
	}
	n := bytes / 4
	if n == 0 || n > uint64(^uint32(0)) {
		return fail("element count overflow")
	}
	if input != output && vkBindingsOverlap(input, output) {
		return fail("partial output overlap")
	}
	groups := uint32((n + 255) / 256)
	push := []uint32{uint32(n)}
	if err := op.kernel.validateBindingsLocked(groups, 1, 1, []vkBufferBinding{input, output}, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{groups, 1, 1}, Tensors: []*VkTensorF32{x, out}, PushWords: push}, nil
}
func (op *VkGELUErfF32) Forward(ctx context.Context, out, x *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x)
	if err != nil {
		return err
	}
	input, err := x.bindingLocked()
	if err != nil {
		return err
	}
	output, err := out.bindingLocked()
	if err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], 1, 1, []vkBufferBinding{input, output}, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
