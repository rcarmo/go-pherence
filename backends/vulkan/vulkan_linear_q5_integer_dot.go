package vulkan

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
)

//go:embed shaders/integer-dot/q8-stripped.spv
var spirv_quant_q8_integer_dot []byte

//go:embed shaders/integer-dot/linear-stripped.spv
var spirv_linear_q5_integer_dot []byte

// VkLinearQ5IntegerDotSet is an explicit experimental arithmetic owner. Original
// Q5 bytes are unchanged; activations and scales are quantised to Q8_1. It is not
// F32-equivalent. Callers must provide finite, bounded activations; there is no
// fallback or implicit device-feature upgrade. Magnitudes above1000, nonfinite
// values or nonzero block maxima below1e-30 yield NaN scale/sum markers; callers
// must reject nonfinite final output. Scratch uses uint32 payloads in
// F32 backing storage, never numerical float uploads/downloads.
type VkLinearQ5IntegerDotSet struct {
	weights *VkLinearQ5Set
	quant   *VkComputeKernel
	f32     *VkComputeKernel // explicit hybrid mode only, shares original storage
}

func NewVkLinearQ5IntegerDotSetStream(ctx context.Context, shapes []VkLinearQ5Shape, read func(context.Context, int) ([]byte, error)) (*VkLinearQ5IntegerDotSet, error) {
	weights, err := newVkLinearQ5SetStream(ctx, shapes, read, true)
	if err != nil {
		if weights != nil {
			return &VkLinearQ5IntegerDotSet{weights: weights}, err
		}
		return nil, err
	}
	owner := &VkLinearQ5IntegerDotSet{weights: weights}
	if err = ctx.Err(); err == nil {
		owner.quant, err = VkKernelCreateIntegerDot(spirv_quant_q8_integer_dot, 2, 4)
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		if closeErr := owner.Close(); closeErr != nil {
			return owner, errors.Join(err, closeErr)
		}
		return nil, err
	}
	return owner, nil
}
func (s *VkLinearQ5IntegerDotSet) StorageBytes() uint64 {
	if s == nil {
		return 0
	}
	return s.weights.StorageBytes()
}
func (s *VkLinearQ5IntegerDotSet) Close() error {
	if s == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if s.f32 != nil && vkUsesKernelLocked(s.f32) {
		return ErrVulkanInFlight
	}
	if s.weights != nil && (vkUsesKernelLocked(s.weights.kernel) || vkUsesBufferLocked(s.weights.storage)) || s.quant != nil && vkUsesKernelLocked(s.quant) {
		return ErrVulkanInFlight
	}
	if s.weights != nil {
		if err := s.weights.closeLocked(); err != nil {
			return err
		}
	}
	if s.f32 != nil {
		if err := s.f32.closeLocked(); err != nil {
			return err
		}
		s.f32 = nil
	}
	if s.quant != nil {
		if err := s.quant.closeLocked(); err != nil {
			return err
		}
		s.quant = nil
	}
	return nil
}

// Stages returns quantisation and projection stages. scratch must contain exactly
// rows*(inDim/32)*9 words and must not overlap any live input/output/bias storage.
func (s *VkLinearQ5IntegerDotSet) Stages(ctx context.Context, index int, out, x, bias, scratch *VkTensorF32) ([]VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return nil, err
	}
	if !vkIntegerDotEnabled || s == nil || s.weights == nil || s.quant == nil || s.quant.closed || s.quant.device != vkDevice || s.quant.pipeline == 0 || s.weights.kernel == nil || s.weights.kernel.closed || s.weights.kernel.device != vkDevice || index < 0 || index >= len(s.weights.views) || s.weights.storage == nil || s.weights.storage.closed || s.weights.storage.device != vkDevice || s.weights.storage.mapped == nil {
		return nil, fmt.Errorf("Q5 integer-dot set closed/disabled/index")
	}
	view := s.weights.views[index]
	if x == nil || out == nil || bias == nil || scratch == nil || x.rank != 2 || out.rank != 2 || bias.rank != 1 || scratch.rank != 1 {
		return nil, fmt.Errorf("Q5 integer-dot tensor rank")
	}
	rows := x.shape[0]
	blocks := rows * (view.inDim / 32)
	if rows < 1 || rows > 16384 || x.shape[1] != view.inDim || out.shape[0] != rows || out.shape[1] != view.outDim || bias.shape[0] != view.outDim || scratch.shape[0] != blocks*9 {
		return nil, fmt.Errorf("Q5 integer-dot shape")
	}
	xb, e := x.bindingLocked()
	if e != nil {
		return nil, e
	}
	yb, e := out.bindingLocked()
	if e != nil {
		return nil, e
	}
	bb, e := bias.bindingLocked()
	if e != nil {
		return nil, e
	}
	qb, e := scratch.bindingLocked()
	if e != nil {
		return nil, e
	}
	for _, b := range []vkBufferBinding{xb, yb, bb} {
		if vkBindingsOverlap(qb, b) {
			return nil, fmt.Errorf("Q5 integer-dot scratch overlap")
		}
	}
	if vkBindingsOverlap(yb, xb) || vkBindingsOverlap(yb, bb) {
		return nil, fmt.Errorf("Q5 integer-dot output overlap")
	}
	wb := vkBufferBinding{buffer: s.weights.storage, offset: view.offset, size: view.size}
	qgroups := [3]uint32{uint32((blocks + 31) / 32), 1, 1}
	lgroups := [3]uint32{uint32((view.outDim + 63) / 64), uint32((rows + 63) / 64), 1}
	if e = s.quant.validateBindingsLocked(qgroups[0], qgroups[1], qgroups[2], []vkBufferBinding{xb, qb}, unsafePushWords([]uint32{uint32(blocks)})); e != nil {
		return nil, e
	}
	if e = s.weights.kernel.validateBindingsLocked(lgroups[0], lgroups[1], lgroups[2], []vkBufferBinding{qb, wb, bb, yb}, unsafePushWords([]uint32{uint32(rows), uint32(view.inDim), uint32(view.outDim)})); e != nil {
		return nil, e
	}
	return []VkF32Stage{{Kernel: s.quant, Groups: qgroups, bindings: []vkBufferBinding{xb, qb}, PushWords: []uint32{uint32(blocks)}}, {Kernel: s.weights.kernel, Groups: lgroups, bindings: []vkBufferBinding{qb, wb, bb, yb}, PushWords: []uint32{uint32(rows), uint32(view.inDim), uint32(view.outDim)}}}, nil
}

// NewVkLinearQ5IntegerDotHybridSetStream adds the ordered F32 activation kernel
// over the same original packed storage. This is an explicit hybrid arithmetic
// owner, not an unsupported-device fallback. Callers choose the kernel at build.
func NewVkLinearQ5IntegerDotHybridSetStream(ctx context.Context, shapes []VkLinearQ5Shape, read func(context.Context, int) ([]byte, error)) (*VkLinearQ5IntegerDotSet, error) {
	owner, err := NewVkLinearQ5IntegerDotSetStream(ctx, shapes, read)
	if err != nil {
		return owner, err
	}
	if err = ctx.Err(); err == nil {
		owner.f32, err = VkKernelCreate(spirv_linear_q5_grouped_f32, 4, 12)
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		if ce := owner.Close(); ce != nil {
			return owner, errors.Join(err, ce)
		}
		return nil, err
	}
	return owner, nil
}

// F32Stage explicitly retains ordered F32 activations/accumulation for one view.
func (s *VkLinearQ5IntegerDotSet) F32Stage(ctx context.Context, index int, out, x, bias *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if s == nil || s.f32 == nil || s.weights == nil || s.weights.storage == nil || index < 0 || index >= len(s.weights.views) {
		return VkF32Stage{}, fmt.Errorf("Q5 hybrid F32 closed/index")
	}
	view := s.weights.views[index]
	op := VkLinearQ5GroupedF32{kernel: s.f32, weight: s.weights.storage, weightValues: view.outDim * view.inDim, inDim: view.inDim, outDim: view.outDim}
	return op.stageLocked(out, x, bias, view.offset, view.size)
}
