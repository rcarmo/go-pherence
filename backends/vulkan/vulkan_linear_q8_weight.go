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
	owner := &VkLinearQ8WeightF32{kernel: kernel, inDim: inDim, outDim: outDim, weightValues: len(weights), packedBytes: packedBytes, scaleOffsetBytes: scaleOffset}
	defer func() {
		if err != nil {
			if closeErr := owner.closeLocked(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Vulkan LinearQ8WeightF32 rollback: %w", closeErr))
			}
		}
	}()
	owner.storage, err = vkBufAllocLocked(int(scaleOffset + scaleBytes))
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

func (op *VkLinearQ8WeightF32) Forward(ctx context.Context, out, x, bias *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if op == nil || op.kernel == nil || op.storage == nil {
		return fmt.Errorf("Vulkan LinearQ8WeightF32: closed/uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if op.kernel.closed || op.storage.closed || op.kernel.device != vkDevice || op.storage.device != vkDevice || op.storage.mapped == nil || op.storage.buf == 0 || op.storage.mem == 0 {
		return ErrVulkanClosed
	}
	xb, err := x.bindingLocked()
	if err != nil {
		return err
	}
	bb, err := bias.bindingLocked()
	if err != nil {
		return err
	}
	ob, err := out.bindingLocked()
	if err != nil {
		return err
	}
	if x.rank != 2 || bias.rank != 1 || out.rank != 2 {
		return fmt.Errorf("Vulkan LinearQ8WeightF32: rank mismatch")
	}
	rows := x.shape[0]
	if rows < 1 || rows > 16384 || x.shape[1] != op.inDim || bias.shape[0] != op.outDim || out.shape[0] != rows || out.shape[1] != op.outDim {
		return fmt.Errorf("Vulkan LinearQ8WeightF32: projection shapes mismatch")
	}
	scaleBytes := uint64(op.outDim * 4)
	bindings := []vkBufferBinding{xb, {buffer: op.storage, size: op.packedBytes}, {buffer: op.storage, offset: op.scaleOffsetBytes, size: scaleBytes}, bb, ob}
	want := [5]uint64{uint64(rows * op.inDim * 4), uint64((op.weightValues + 3) / 4 * 4), scaleBytes, uint64(op.outDim * 4), uint64(rows * op.outDim * 4)}
	for i, binding := range bindings {
		if binding.size != want[i] {
			return fmt.Errorf("Vulkan LinearQ8WeightF32: shape/storage size mismatch")
		}
		if i != 4 && vkBindingsOverlap(binding, ob) {
			return fmt.Errorf("Vulkan LinearQ8WeightF32: output overlaps input")
		}
	}
	groups := [3]uint32{(uint32(op.outDim) + 15) / 16, (uint32(rows) + 15) / 16, 1}
	push := []uint32{uint32(rows), uint32(op.inDim), uint32(op.outDim)}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], 1, bindings, unsafePushWords(push)); err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, groups[0], groups[1], 1, bindings, unsafePushWords(push))
	runtime.KeepAlive(push)
	runtime.KeepAlive(op)
	return err
}
