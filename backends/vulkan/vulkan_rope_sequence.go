package vulkan

import (
	"context"
	"fmt"
	"math"
	"runtime"
)

// VkRoPESequenceF32 rotates the two halves of each [row,head,headDim]
// vector in place. The frequency table is [row,rotHalf,cos/sin]; Q and K
// may use the same read-only table. This is a plan-compatible device stage.
// Callers must provide finite inputs and call Close after all plans using it.
type VkRoPESequenceF32 struct{ kernel *VkComputeKernel }

func NewVkRoPESequenceF32(ctx context.Context) (*VkRoPESequenceF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	kernel, err := vkKernelCreateLocked(spirv_rope_sequence_f32, 2, 16)
	if err != nil {
		return nil, err
	}
	return &VkRoPESequenceF32{kernel: kernel}, nil
}

func (op *VkRoPESequenceF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

func (op *VkRoPESequenceF32) Stage(ctx context.Context, x, freqs *VkTensorF32, heads int) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(x, freqs, heads)
}

func (op *VkRoPESequenceF32) stageLocked(x, freqs *VkTensorF32, heads int) (VkF32Stage, error) {
	fail := func(reason string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan RoPESequenceF32: %s", reason)
	}
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	if x == nil || freqs == nil {
		return fail("nil tensor")
	}
	input, err := x.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	frequency, err := freqs.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	if x.rank != 2 || freqs.rank != 2 || heads < 1 || heads > 32 {
		return fail("rank or head count")
	}
	rows, width, half := x.shape[0], x.shape[1], freqs.shape[1]/2
	if rows < 1 || rows > 4096 || width < 1 || width > 2048 || width%heads != 0 || half < 1 || freqs.shape[0] != rows || freqs.shape[1] != half*2 {
		return fail("sequence or frequency shape")
	}
	dim := width / heads
	if dim < 2 || dim > 64 || half > dim/2 {
		return fail("head or rotation width")
	}
	// Bound all shader uint32 products before conversion or dispatch.
	if uint64(rows)*uint64(width) > math.MaxUint32 || uint64(rows)*uint64(half)*2 > math.MaxUint32 || uint64(rows)*uint64(heads)*uint64(half) > math.MaxUint32 {
		return fail("element count overflow")
	}
	if input.size != uint64(rows)*uint64(width)*4 || frequency.size != uint64(rows)*uint64(half)*8 || vkBindingsOverlap(input, frequency) {
		return fail("storage size or overlapping frequency table")
	}
	groups := uint32((uint64(rows)*uint64(heads)*uint64(half) + 255) / 256)
	push := []uint32{uint32(rows), uint32(heads), uint32(dim), uint32(half)}
	if err := op.kernel.validateBindingsLocked(groups, 1, 1, []vkBufferBinding{input, frequency}, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{groups, 1, 1}, Tensors: []*VkTensorF32{x, freqs}, PushWords: push}, nil
}

func (op *VkRoPESequenceF32) Forward(ctx context.Context, x, freqs *VkTensorF32, heads int) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(x, freqs, heads)
	if err != nil {
		return err
	}
	input, err := x.bindingLocked()
	if err != nil {
		return err
	}
	frequency, err := freqs.bindingLocked()
	if err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], 1, 1, []vkBufferBinding{input, frequency}, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
