package vulkan

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// VkLinearQ5Matrix contains original row-major GGML Q5_0 blocks. No requantisation.
type VkLinearQ5Matrix struct {
	Blocks        []byte
	OutDim, InDim int
}
type vkQ5View struct {
	outDim, inDim int
	offset, size  uint64
}

// VkLinearQ5Set owns one pipeline and one aligned immutable packed allocation.
// Close plans using its checked stages before closing this owner.
type VkLinearQ5Set struct {
	kernel  *VkComputeKernel
	storage *VkBuf
	views   []vkQ5View
	bytes   uint64
}

func prepareQ5Set(ctx context.Context, matrices []VkLinearQ5Matrix, alignment uint64) ([]vkQ5View, [][]uint32, uint64, error) {
	if ctx == nil {
		return nil, nil, 0, fmt.Errorf("Q5 set: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, 0, err
	}
	if len(matrices) < 1 || len(matrices) > 512 {
		return nil, nil, 0, fmt.Errorf("Q5 set: count")
	}
	views := make([]vkQ5View, len(matrices))
	packed := make([][]uint32, len(matrices))
	var total uint64
	for i, m := range matrices {
		if m.OutDim < 1 || m.OutDim > 16384 || m.InDim < 32 || m.InDim > 16384 || m.InDim%32 != 0 || len(m.Blocks) != m.OutDim*(m.InDim/32)*22 {
			return nil, nil, 0, fmt.Errorf("Q5 set: matrix%d shape", i)
		}
		offset, err := alignQ8Offset(total, alignment)
		if err != nil {
			return nil, nil, 0, err
		}
		size := uint64(len(m.Blocks)/22) * 24
		if offset > 4<<30 || size > (4<<30)-offset {
			return nil, nil, 0, fmt.Errorf("Q5 set: total budget")
		}
		total = offset + size
		views[i] = vkQ5View{m.OutDim, m.InDim, offset, size}
	}
	// Admit the aggregate budget before allocating widened temporary blocks.
	for i, m := range matrices {
		words, err := packLinearQ5Blocks(ctx, m.Blocks)
		if err != nil {
			return nil, nil, 0, err
		}
		packed[i] = words
	}
	if total > uint64(^uint(0)>>1) {
		return nil, nil, 0, fmt.Errorf("Q5 set: host extent")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, 0, err
	}
	return views, packed, total, nil
}
func NewVkLinearQ5Set(ctx context.Context, matrices []VkLinearQ5Matrix) (result *VkLinearQ5Set, err error) {
	if err = vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	views, packed, total, err := prepareQ5Set(ctx, matrices, vkLimits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_q5_grouped_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	owner := &VkLinearQ5Set{kernel: kernel, views: views, bytes: total}
	defer func() {
		if err != nil {
			if e := owner.closeLocked(); e != nil {
				result = owner
				err = errors.Join(err, e)
			}
		}
	}()
	owner.storage, err = vkBufAllocLocked(int(total))
	if err != nil {
		return nil, err
	}
	for i, v := range views {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		copy(unsafe.Slice((*uint32)(unsafe.Add(owner.storage.mapped, uintptr(v.offset))), len(packed[i])), packed[i])
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	runtime.KeepAlive(owner.storage)
	return owner, nil
}
func (s *VkLinearQ5Set) StorageBytes() uint64 {
	if s == nil {
		return 0
	}
	return s.bytes
}
func (s *VkLinearQ5Set) Stage(ctx context.Context, index int, out, x, bias *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if s == nil || s.kernel == nil || s.storage == nil || index < 0 || index >= len(s.views) {
		return VkF32Stage{}, fmt.Errorf("Q5 set: closed/index")
	}
	v := s.views[index]
	op := &VkLinearQ5GroupedF32{kernel: s.kernel, weight: s.storage, inDim: v.inDim, outDim: v.outDim, weightValues: v.inDim * v.outDim}
	return op.stageLocked(out, x, bias, v.offset, v.size)
}
func (s *VkLinearQ5Set) Close() error {
	if s == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	return s.closeLocked()
}
func (s *VkLinearQ5Set) closeLocked() error {
	if s == nil {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if s.storage != nil && vkUsesBufferLocked(s.storage) || s.kernel != nil && vkUsesKernelLocked(s.kernel) {
		return ErrVulkanInFlight
	}
	var failures []error
	if s.storage != nil {
		if err := s.storage.freeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			s.storage = nil
		}
	}
	if s.kernel != nil {
		if err := s.kernel.closeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			s.kernel = nil
		}
	}
	if len(failures) == 0 {
		s.views = nil
	}
	return errors.Join(failures...)
}
