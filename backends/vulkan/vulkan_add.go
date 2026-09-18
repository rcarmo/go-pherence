package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkAddF32 provides checked arena/plan addition using the unchanged vec_add_f32
// shader. Shapes must match exactly, rank1..8; no broadcasting. Exact output
// alias with either/both inputs is allowed, partial overlap rejected. Caller
// ensures finite representable sums. No data scan or hidden fallback.
type VkAddF32 struct{ kernel *VkComputeKernel }

func NewVkAddF32(ctx context.Context) (*VkAddF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_vec_add_f32, 3, 4)
	if err != nil {
		return nil, err
	}
	return &VkAddF32{kernel: k}, nil
}
func (op *VkAddF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage checks addition invariants at creation. Returned generic stages are
// mutable; plans do not recheck these operator-specific shape/alias rules.
// Callers modifying a stage assume responsibility for shader safety.
func (op *VkAddF32) Stage(ctx context.Context, out, a, b *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, a, b)
}
func (op *VkAddF32) stageLocked(out, a, b *VkTensorF32) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan AddF32: %s", s) }
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{a, b, out}
	bindings := make([]vkBufferBinding, 3)
	for i, t := range tensors {
		v, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = v
	}
	if a.rank < 1 || a.rank > 8 || b.rank != a.rank || out.rank != a.rank {
		return fail("rank mismatch")
	}
	for i := 0; i < a.rank; i++ {
		if a.shape[i] != b.shape[i] || a.shape[i] != out.shape[i] {
			return fail("shape mismatch")
		}
	}
	size, err := vkF32ShapeBytes(a.shape[:a.rank])
	if err != nil {
		return VkF32Stage{}, err
	}
	n := size / 4
	if n == 0 || n > uint64(^uint32(0)) {
		return fail("element count overflow")
	}
	for i, v := range bindings {
		if v.size != size {
			return fail("shape/storage mismatch")
		}
		if i < 2 && v != bindings[2] && vkBindingsOverlap(v, bindings[2]) {
			return fail("partial output overlap")
		}
	}
	groups := uint32((n + 255) / 256)
	push := []uint32{uint32(n)}
	if err := op.kernel.validateBindingsLocked(groups, 1, 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{groups, 1, 1}, Tensors: tensors, PushWords: push}, nil
}
func (op *VkAddF32) Forward(ctx context.Context, out, a, b *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	s, err := op.stageLocked(out, a, b)
	if err != nil {
		return err
	}
	bindings := make([]vkBufferBinding, 3)
	for i, t := range s.Tensors {
		bindings[i], err = t.bindingLocked()
		if err != nil {
			return err
		}
	}
	err = op.kernel.dispatchBindingsLocked(ctx, s.Groups[0], 1, 1, bindings, unsafePushWords(s.PushWords))
	runtime.KeepAlive(s.PushWords)
	return err
}
