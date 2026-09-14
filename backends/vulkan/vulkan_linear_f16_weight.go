package vulkan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
)

// VkLinearF16WeightF32 owns one immutable row-major IEEE-F16 weight matrix and
// a mixed-storage projection kernel. Inputs, bias, accumulation and output stay
// F32. F16 values are packed little-half-first into uint32 storage and widened
// in the shader. This uses no optional Vulkan 16-bit storage/compute feature.
// It is a standalone opt-in operator and cannot be placed in VkF32Plan.
// Copies share kernel/buffer closure; callers must exclude Forward from Close.
type VkLinearF16WeightF32 struct {
	kernel        *VkComputeKernel
	weight        *VkBuf
	inDim, outDim int
	weightValues  int
}

// NewVkLinearF16WeightF32 validates, narrows and owns weights[N,K]. Every input
// must be finite and remain finite after IEEE-F16 rounding. Odd element counts
// use a zero upper half in the last uint32 word. Construction allocates one
// dedicated host-visible/coherent weight buffer after creating the kernel.
func NewVkLinearF16WeightF32(ctx context.Context, weights []float32, outDim, inDim int) (result *VkLinearF16WeightF32, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Vulkan LinearF16WeightF32: nil context")
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if outDim < 1 || outDim > 16384 || inDim < 1 || inDim > 16384 || outDim > int(^uint(0)>>1)/inDim || len(weights) != outDim*inDim {
		return nil, fmt.Errorf("Vulkan LinearF16WeightF32: invalid weight shape")
	}
	packed, err := packLinearF16Weights(weights)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_f16_weight_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	owner := &VkLinearF16WeightF32{kernel: kernel, inDim: inDim, outDim: outDim, weightValues: len(weights)}
	defer func() {
		if err != nil {
			if closeErr := owner.closeLocked(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Vulkan LinearF16WeightF32 rollback: %w", closeErr))
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

func packLinearF16Weights(weights []float32) ([]uint32, error) {
	if len(weights) == 0 {
		return nil, fmt.Errorf("Vulkan LinearF16WeightF32: empty weights")
	}
	packed := make([]uint32, (len(weights)+1)/2)
	for i, value := range weights {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("Vulkan LinearF16WeightF32: nonfinite weight")
		}
		bits := half.F32ToF16(value)
		if bits&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("Vulkan LinearF16WeightF32: weight outside finite F16")
		}
		packed[i/2] |= uint32(bits) << (16 * uint(i&1))
	}
	return packed, nil
}

func (op *VkLinearF16WeightF32) Close() error {
	if op == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	return op.closeLocked()
}

func (op *VkLinearF16WeightF32) closeLocked() error {
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
func (op *VkLinearF16WeightF32) Forward(ctx context.Context, out, x, bias *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if op == nil || op.kernel == nil || op.weight == nil {
		return fmt.Errorf("Vulkan LinearF16WeightF32: closed/uninitialized operator")
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
		return fmt.Errorf("Vulkan LinearF16WeightF32: rank mismatch")
	}
	rows := x.shape[0]
	if rows < 1 || rows > 16384 || x.shape[1] != op.inDim || bias.shape[0] != op.outDim || out.shape[0] != rows || out.shape[1] != op.outDim {
		return fmt.Errorf("Vulkan LinearF16WeightF32: projection shapes mismatch")
	}
	packedBytes := uint64((op.weightValues + 1) / 2 * 4)
	bindings := []vkBufferBinding{xb, {buffer: op.weight, size: packedBytes}, bb, ob}
	want := [4]uint64{uint64(rows * op.inDim * 4), packedBytes, uint64(op.outDim * 4), uint64(rows * op.outDim * 4)}
	for i, binding := range bindings {
		if binding.size != want[i] {
			return fmt.Errorf("Vulkan LinearF16WeightF32: shape/storage size mismatch")
		}
		if i != 3 && vkBindingsOverlap(binding, ob) {
			return fmt.Errorf("Vulkan LinearF16WeightF32: output overlaps input")
		}
	}
	groups := [3]uint32{(uint32(op.outDim) + 15) / 16, (uint32(rows) + 15) / 16, 1}
	push := []uint32{uint32(rows), uint32(op.inDim), uint32(op.outDim)}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], groups[2], bindings, unsafePushWords(push)); err != nil {
		return err
	}
	err = op.kernel.dispatchBindingsLocked(ctx, groups[0], groups[1], 1, bindings, unsafePushWords(push))
	runtime.KeepAlive(push)
	runtime.KeepAlive(op)
	return err
}
