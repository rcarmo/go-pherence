package vulkan

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// VkLinearQ8WeightMatrix describes one row-major F32 matrix to quantise.
type VkLinearQ8WeightMatrix struct {
	Weights       []float32
	OutDim, InDim int
}

type vkLinearQ8WeightView struct {
	inDim, outDim, weightValues  int
	packedOffset, packedBytes    uint64
	scaleOffsetBytes, scaleBytes uint64
}

// VkLinearQ8WeightSet owns one Q8 pipeline and one aligned packed allocation for
// several immutable matrices. It is intended for model graphs with many linear
// projections. Copies share closure; every plan using a view must close first.
type VkLinearQ8WeightSet struct {
	kernel       *VkComputeKernel
	storage      *VkBuf
	storageBytes uint64
	views        []vkLinearQ8WeightView
}

func alignQ8Offset(value, alignment uint64) (uint64, error) {
	if alignment < 4 {
		alignment = 4
	}
	if alignment&(alignment-1) != 0 || value > ^uint64(0)-(alignment-1) {
		return 0, fmt.Errorf("Vulkan LinearQ8WeightSet: invalid alignment/extent")
	}
	return (value + alignment - 1) &^ (alignment - 1), nil
}

func prepareLinearQ8WeightSet(ctx context.Context, matrices []VkLinearQ8WeightMatrix, alignment uint64) ([]vkLinearQ8WeightView, [][]uint32, [][]float32, uint64, error) {
	if ctx == nil {
		return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: nil context")
	}
	if len(matrices) < 1 || len(matrices) > 512 {
		return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: matrix count must be1..512")
	}
	alignment = max(uint64(4), alignment)
	views := make([]vkLinearQ8WeightView, len(matrices))
	packedRows := make([][]uint32, len(matrices))
	scaleRows := make([][]float32, len(matrices))
	var total uint64
	for i, matrix := range matrices {
		if err := ctx.Err(); err != nil {
			return nil, nil, nil, 0, err
		}
		if matrix.OutDim < 1 || matrix.OutDim > 16384 || matrix.InDim < 1 || matrix.InDim > 16384 || matrix.OutDim > int(^uint(0)>>1)/matrix.InDim || len(matrix.Weights) != matrix.OutDim*matrix.InDim {
			return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: invalid matrix%d shape", i)
		}
		packed, scales, err := packLinearQ8Weights(matrix.Weights, matrix.OutDim, matrix.InDim)
		if err != nil {
			return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: matrix%d: %w", i, err)
		}
		packedOffset, err := alignQ8Offset(total, alignment)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		packedBytes := uint64(len(packed) * 4)
		scaleOffset, err := alignQ8Offset(packedOffset+packedBytes, alignment)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		scaleBytes := uint64(len(scales) * 4)
		if scaleOffset > ^uint64(0)-scaleBytes {
			return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: storage overflow")
		}
		views[i] = vkLinearQ8WeightView{inDim: matrix.InDim, outDim: matrix.OutDim, weightValues: len(matrix.Weights), packedOffset: packedOffset, packedBytes: packedBytes, scaleOffsetBytes: scaleOffset, scaleBytes: scaleBytes}
		packedRows[i], scaleRows[i] = packed, scales
		total = scaleOffset + scaleBytes
	}
	if total > uint64(^uint(0)>>1) {
		return nil, nil, nil, 0, fmt.Errorf("Vulkan LinearQ8WeightSet: storage exceeds host int")
	}
	return views, packedRows, scaleRows, total, nil
}

func NewVkLinearQ8WeightSet(ctx context.Context, matrices []VkLinearQ8WeightMatrix) (result *VkLinearQ8WeightSet, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Vulkan LinearQ8WeightSet: nil context")
	}
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	views, packedRows, scaleRows, total, err := prepareLinearQ8WeightSet(ctx, matrices, vkLimits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_q8_weight_f32, 5, 12)
	if err != nil {
		return nil, err
	}
	owner := &VkLinearQ8WeightSet{kernel: kernel, views: views, storageBytes: total}
	defer func() {
		if err != nil {
			if closeErr := owner.closeLocked(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Vulkan LinearQ8WeightSet rollback: %w", closeErr))
			}
		}
	}()
	owner.storage, err = vkBufAllocLocked(int(total))
	if err != nil {
		return nil, err
	}
	for i, view := range views {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		copy(unsafe.Slice((*uint32)(unsafe.Add(owner.storage.mapped, uintptr(view.packedOffset))), len(packedRows[i])), packedRows[i])
		copy(unsafe.Slice((*float32)(unsafe.Add(owner.storage.mapped, uintptr(view.scaleOffsetBytes))), len(scaleRows[i])), scaleRows[i])
	}
	runtime.KeepAlive(owner.storage)
	return owner, nil
}

func (set *VkLinearQ8WeightSet) Stage(ctx context.Context, index int, out, x, bias *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if set == nil || set.kernel == nil || set.storage == nil || index < 0 || index >= len(set.views) {
		return VkF32Stage{}, fmt.Errorf("Vulkan LinearQ8WeightSet: closed/uninitialized set or invalid index")
	}
	return linearQ8StageLocked(set.kernel, set.storage, set.views[index], out, x, bias)
}
func (set *VkLinearQ8WeightSet) Count() int {
	if set == nil {
		return 0
	}
	return len(set.views)
}
func (set *VkLinearQ8WeightSet) StorageBytes() uint64 {
	if set == nil {
		return 0
	}
	return set.storageBytes
}
func (set *VkLinearQ8WeightSet) Close() error {
	if set == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	return set.closeLocked()
}
func (set *VkLinearQ8WeightSet) closeLocked() error {
	if set == nil {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if set.storage != nil && vkUsesBufferLocked(set.storage) || set.kernel != nil && vkUsesKernelLocked(set.kernel) {
		return ErrVulkanInFlight
	}
	var failures []error
	if set.storage != nil {
		if err := set.storage.freeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			set.storage = nil
		}
	}
	if set.kernel != nil {
		if err := set.kernel.closeLocked(); err != nil {
			failures = append(failures, err)
		} else {
			set.kernel = nil
		}
	}
	if len(failures) == 0 {
		set.views = nil
	}
	return errors.Join(failures...)
}
