package vulkan

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// VkLinearQ5GroupedF32 owns immutable original GGML Q5_0 blocks (32 values,
// 22 bytes/block), losslessly repacked into aligned 24-byte blocks. Eight lanes
// per block widen stored values into F32 shared memory. Input, bias, ordered
// accumulation and output remain F32. No quantisation or F16 activation rounding
// is performed. Standalone opt-in; not admitted to VkF32Plan or serving defaults.
// Copies share native closure; callers must exclude Forward from Close.
type VkLinearQ5GroupedF32 struct {
	kernel        *VkComputeKernel
	weight        *VkBuf
	inDim, outDim int
	weightValues  int
}

// NewVkLinearQ5GroupedF32 copies and validates row-major original Q5_0 blocks.
// K must be divisible by 32. F16 scales must be finite; raw length must exactly
// match N*K/32*22. Packing preserves every source bit and adds two zero bytes
// after each scale. The caller may release raw after successful construction.
func NewVkLinearQ5GroupedF32(ctx context.Context, raw []byte, outDim, inDim int) (result *VkLinearQ5GroupedF32, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Vulkan LinearQ5GroupedF32: nil context")
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if outDim < 1 || outDim > 16384 || inDim < 1 || inDim > 16384 || outDim > int(^uint(0)>>1)/inDim || inDim%32 != 0 || len(raw) != outDim*(inDim/32)*22 {
		return nil, fmt.Errorf("Vulkan LinearQ5GroupedF32: invalid weight shape")
	}
	packed, err := packLinearQ5Blocks(ctx, raw)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_q5_grouped_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	owner := &VkLinearQ5GroupedF32{kernel: kernel, inDim: inDim, outDim: outDim, weightValues: outDim * inDim}
	defer func() {
		if err != nil {
			if closeErr := owner.closeLocked(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Vulkan LinearQ5GroupedF32 rollback: %w", closeErr))
			}
		}
	}()
	owner.weight, err = vkBufAllocLocked(len(packed) * 4)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	dst := unsafe.Slice((*uint32)(owner.weight.mapped), len(packed))
	copy(dst, packed)
	runtime.KeepAlive(owner.weight)
	return owner, nil
}

func packLinearQ5Blocks(ctx context.Context, raw []byte) ([]uint32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Vulkan LinearQ5GroupedF32: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(raw) == 0 || len(raw)%22 != 0 || len(raw)/22 > 16384*(16384/32) {
		return nil, fmt.Errorf("Vulkan LinearQ5GroupedF32: block boundary/budget")
	}
	words := make([]uint32, len(raw)/22*6)
	for b := 0; b < len(raw)/22; b++ {
		if b%1024 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		src := raw[b*22:][:22]
		scale := binary.LittleEndian.Uint16(src)
		if scale&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("Vulkan LinearQ5GroupedF32: nonfinite scale")
		}
		words[b*6] = uint32(scale)
		words[b*6+1] = binary.LittleEndian.Uint32(src[2:6])
		for j := 0; j < 4; j++ {
			words[b*6+2+j] = binary.LittleEndian.Uint32(src[6+j*4:])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return words, nil
}

func (op *VkLinearQ5GroupedF32) Close() error {
	if op == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	return op.closeLocked()
}

func (op *VkLinearQ5GroupedF32) closeLocked() error {
	if op == nil {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	// Preflight the complete composite owner before freeing either resource.
	if op.weight != nil && vkUsesBufferLocked(op.weight) || op.kernel != nil && vkUsesKernelLocked(op.kernel) {
		return ErrVulkanInFlight
	}
	var failures []error
	// Reverse construction order. A failed close leaves each owner reachable.
	if op.weight != nil {
		if err := op.weight.freeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			op.weight = nil
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

// Forward validates X[M,K], Bias[N], Out[M,N]. Output cannot overlap X or Bias.
// The immutable dedicated weight buffer cannot alias caller arena tensors.
func (op *VkLinearQ5GroupedF32) Forward(ctx context.Context, out, x, bias *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if op == nil || op.kernel == nil || op.weight == nil {
		return fmt.Errorf("Vulkan LinearQ5GroupedF32: closed/uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if op.kernel.closed || op.weight.closed || op.kernel.device != vkDevice || op.weight.device != vkDevice || op.weight.mapped == nil || op.weight.buf == 0 || op.weight.mem == 0 {
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
		return fmt.Errorf("Vulkan LinearQ5GroupedF32: rank mismatch")
	}
	rows := x.shape[0]
	if rows < 1 || rows > 16384 || x.shape[1] != op.inDim || bias.shape[0] != op.outDim || out.shape[0] != rows || out.shape[1] != op.outDim {
		return fmt.Errorf("Vulkan LinearQ5GroupedF32: projection shapes mismatch")
	}
	packedBytes := uint64(op.weightValues / 32 * 24)
	bindings := []vkBufferBinding{xb, {buffer: op.weight, size: packedBytes}, bb, ob}
	want := [4]uint64{uint64(rows * op.inDim * 4), packedBytes, uint64(op.outDim * 4), uint64(rows * op.outDim * 4)}
	for i, binding := range bindings {
		if binding.size != want[i] {
			return fmt.Errorf("Vulkan LinearQ5GroupedF32: shape/storage size mismatch")
		}
		if i != 3 && vkBindingsOverlap(binding, ob) {
			return fmt.Errorf("Vulkan LinearQ5GroupedF32: output overlaps input")
		}
	}
	groups := [3]uint32{(uint32(op.outDim) + 63) / 64, (uint32(rows) + 63) / 64, 1}
	push := []uint32{uint32(rows), uint32(op.inDim), uint32(op.outDim)}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], groups[2], bindings, unsafePushWords(push)); err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, groups[0], groups[1], 1, bindings, unsafePushWords(push))
	runtime.KeepAlive(push)
	runtime.KeepAlive(op)
	return err
}
