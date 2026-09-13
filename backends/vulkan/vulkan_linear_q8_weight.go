package vulkan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// VkLinearQ8WeightF32 owns a symmetric per-output-row Q8 weight matrix. Four
// signed bytes are packed little-byte-first per uint32; each output row has one
// F32 scale max(abs(row))/127. X, bias, accumulation and output remain F32.
// This standalone candidate requires no optional integer-dot Vulkan feature and
// cannot enter VkF32Plan. Copies share closure; exclude Forward from Close.
type VkLinearQ8WeightF32 struct {
	kernel                        *VkComputeKernel
	storage                       *VkBuf
	inDim, outDim, weightValues   int
	packedBytes, scaleOffsetBytes uint64
	storageBytes                  uint64
}

func packLinearQ8Weights(weights []float32, outDim, inDim int) ([]uint32, []float32, error) {
	if outDim < 1 || inDim < 1 || outDim > int(^uint(0)>>1)/inDim || len(weights) != outDim*inDim {
		return nil, nil, fmt.Errorf("Vulkan LinearQ8WeightF32: invalid weight shape")
	}
	packed := make([]uint32, (len(weights)+3)/4)
	scales := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		base := row * inDim
		maximum := float32(0)
		for _, value := range weights[base : base+inDim] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, nil, fmt.Errorf("Vulkan LinearQ8WeightF32: nonfinite weight")
			}
			maximum = max(maximum, float32(math.Abs(float64(value))))
		}
		scale := maximum / 127
		scales[row] = scale
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		for column, value := range weights[base : base+inDim] {
			q := int(math.RoundToEven(float64(value * inverse)))
			q = min(127, max(-127, q))
			index := base + column
			packed[index/4] |= uint32(uint8(int8(q))) << (8 * uint(index&3))
		}
	}
	return packed, scales, nil
}

func NewVkLinearQ8WeightF32(ctx context.Context, weights []float32, outDim, inDim int) (result *VkLinearQ8WeightF32, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Vulkan LinearQ8WeightF32: nil context")
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if outDim < 1 || outDim > 16384 || inDim < 1 || inDim > 16384 {
		return nil, fmt.Errorf("Vulkan LinearQ8WeightF32: dimensions must be1..16384")
	}
	packed, scales, err := packLinearQ8Weights(weights, outDim, inDim)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_q8_weight_f32, 5, 12)
	if err != nil {
		return nil, err
	}
	alignment := max(uint64(4), vkLimits.StorageBufferOffsetAlignment)
	packedBytes := uint64(len(packed) * 4)
	pad := (alignment - packedBytes%alignment) % alignment
	scaleOffset := packedBytes + pad
	scaleBytes := uint64(len(scales) * 4)
	if scaleOffset > uint64(^uint(0)>>1) || scaleBytes > uint64(^uint(0)>>1)-scaleOffset {
		_ = kernel.closeLocked()
		return nil, fmt.Errorf("Vulkan LinearQ8WeightF32: storage overflow")
	}
	owner := &VkLinearQ8WeightF32{kernel: kernel, inDim: inDim, outDim: outDim, weightValues: len(weights), packedBytes: packedBytes, scaleOffsetBytes: scaleOffset, storageBytes: scaleOffset + scaleBytes}
	defer func() {
		if err != nil {
			if closeErr := owner.closeLocked(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Vulkan LinearQ8WeightF32 rollback: %w", closeErr))
			}
		}
	}()
	owner.storage, err = vkBufAllocLocked(int(owner.storageBytes))
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	copy(unsafe.Slice((*uint32)(owner.storage.mapped), len(packed)), packed)
	copy(unsafe.Slice((*float32)(unsafe.Add(owner.storage.mapped, uintptr(scaleOffset))), len(scales)), scales)
	runtime.KeepAlive(owner.storage)
	return owner, nil
}

func (op *VkLinearQ8WeightF32) Close() error {
	if op == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	return op.closeLocked()
}
func (op *VkLinearQ8WeightF32) closeLocked() error {
	if op == nil {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if op.storage != nil && vkUsesBufferLocked(op.storage) || op.kernel != nil && vkUsesKernelLocked(op.kernel) {
		return ErrVulkanInFlight
	}
	var failures []error
	if op.storage != nil {
		if err := op.storage.freeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			op.storage = nil
		}
	}
	if op.kernel != nil {
		if err := op.kernel.closeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			op.kernel = nil
		}
	}
	return errors.Join(failures...)
}

// Stage validates a Q8 projection for inclusion in VkF32Plan. The returned
// stage privately retains the packed-weight and scale ranges; the operator must
// outlive the plan and cannot Close while a submission is pending.
func (op *VkLinearQ8WeightF32) Stage(ctx context.Context, out, x, bias *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, bias)
}

func (op *VkLinearQ8WeightF32) stageLocked(out, x, bias *VkTensorF32) (VkF32Stage, error) {
	if op == nil || op.kernel == nil || op.storage == nil {
		return VkF32Stage{}, fmt.Errorf("Vulkan LinearQ8WeightF32: closed/uninitialized operator")
	}
	view := vkLinearQ8WeightView{inDim: op.inDim, outDim: op.outDim, weightValues: op.weightValues, packedBytes: op.packedBytes, scaleOffsetBytes: op.scaleOffsetBytes, scaleBytes: uint64(op.outDim * 4)}
	return linearQ8StageLocked(op.kernel, op.storage, view, out, x, bias)
}

func linearQ8StageLocked(kernel *VkComputeKernel, storage *VkBuf, view vkLinearQ8WeightView, out, x, bias *VkTensorF32) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) {
		return VkF32Stage{}, fmt.Errorf("Vulkan LinearQ8WeightF32: %s", s)
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	if kernel == nil || storage == nil || kernel.closed || storage.closed || kernel.device != vkDevice || storage.device != vkDevice || storage.mapped == nil || storage.buf == 0 || storage.mem == 0 {
		return VkF32Stage{}, ErrVulkanClosed
	}
	xb, err := x.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	bb, err := bias.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	ob, err := out.bindingLocked()
	if err != nil {
		return VkF32Stage{}, err
	}
	if x.rank != 2 || bias.rank != 1 || out.rank != 2 {
		return fail("rank mismatch")
	}
	rows := x.shape[0]
	if rows < 1 || rows > 16384 || x.shape[1] != view.inDim || bias.shape[0] != view.outDim || out.shape[0] != rows || out.shape[1] != view.outDim {
		return fail("projection shapes mismatch")
	}
	bindings := []vkBufferBinding{xb, {buffer: storage, offset: view.packedOffset, size: view.packedBytes}, {buffer: storage, offset: view.scaleOffsetBytes, size: view.scaleBytes}, bb, ob}
	want := [5]uint64{uint64(rows * view.inDim * 4), uint64((view.weightValues + 3) / 4 * 4), uint64(view.outDim * 4), uint64(view.outDim * 4), uint64(rows * view.outDim * 4)}
	for i, binding := range bindings {
		if binding.size != want[i] {
			return fail("shape/storage size mismatch")
		}
		if i != 4 && vkBindingsOverlap(binding, ob) {
			return fail("output overlaps input")
		}
	}
	groups := [3]uint32{(uint32(view.outDim) + 31) / 32, (uint32(rows) + 31) / 32, 1}
	push := []uint32{uint32(rows), uint32(view.inDim), uint32(view.outDim)}
	if err := kernel.validateBindingsLocked(groups[0], groups[1], 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: kernel, Groups: groups, PushWords: push, bindings: bindings}, nil
}

func (op *VkLinearQ8WeightF32) Forward(ctx context.Context, out, x, bias *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x, bias)
	if err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], stage.Groups[1], stage.Groups[2], stage.bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	runtime.KeepAlive(op)
	return err
}

// StorageBytes reports the immutable packed-weight allocation extent, including
// descriptor-alignment padding and F32 row scales.
func (op *VkLinearQ8WeightF32) StorageBytes() uint64 {
	if op == nil {
		return 0
	}
	return op.storageBytes
}
