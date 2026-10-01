package vulkan

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// VkLinearQ5Shape describes one row-major original Q5_0 matrix.
type VkLinearQ5Shape struct{ OutDim, InDim int }

func describeQ5Stream(ctx context.Context, shapes []VkLinearQ5Shape, alignment uint64) ([]vkQ5View, uint64, error) {
	if ctx == nil {
		return nil, 0, fmt.Errorf("Q5 stream: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if len(shapes) < 1 || len(shapes) > 512 {
		return nil, 0, fmt.Errorf("Q5 stream: matrix count")
	}
	views := make([]vkQ5View, len(shapes))
	var total uint64
	for i, s := range shapes {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		if s.OutDim < 1 || s.OutDim > 16384 || s.InDim < 32 || s.InDim > 16384 || s.InDim%32 != 0 {
			return nil, 0, fmt.Errorf("Q5 stream: matrix%d shape", i)
		}
		offset, err := alignQ8Offset(total, alignment)
		if err != nil {
			return nil, 0, err
		}
		size := uint64(s.OutDim) * uint64(s.InDim/32) * 24
		if offset > 4<<30 || size > (4<<30)-offset {
			return nil, 0, fmt.Errorf("Q5 stream: total budget")
		}
		total = offset + size
		views[i] = vkQ5View{s.OutDim, s.InDim, offset, size}
	}
	if total > uint64(^uint(0)>>1) {
		return nil, 0, fmt.Errorf("Q5 stream: host extent")
	}
	return views, total, nil
}

// packQ5Into writes directly into final unpublished storage. It may leave a
// partial destination on error; constructor rollback owns/discards that storage.
// raw and dst must not overlap or mutate concurrently. No raw view is retained.
func packQ5Into(ctx context.Context, raw []byte, dst []uint32) error {
	if ctx == nil {
		return fmt.Errorf("Q5 stream: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(raw) == 0 || len(raw)%22 != 0 || len(raw)/22 > 16384*(16384/32) || len(dst) != len(raw)/22*6 {
		return fmt.Errorf("Q5 stream: extent")
	}
	for b := 0; b < len(raw)/22; b++ {
		if b%1024 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		block := raw[b*22:][:22]
		scale := binary.LittleEndian.Uint16(block)
		if scale&0x7c00 == 0x7c00 {
			return fmt.Errorf("Q5 stream: nonfinite scale")
		}
		out := dst[b*6:][:6]
		out[0] = uint32(scale)
		out[1] = binary.LittleEndian.Uint32(block[2:6])
		for j := 0; j < 4; j++ {
			out[j+2] = binary.LittleEndian.Uint32(block[6+j*4:])
		}
	}
	return ctx.Err()
}

// NewVkLinearQ5SetStream admits geometry and native allocation before requesting
// each matrix once. read must return original Q5_0 bytes of the exact extent;
// they are synchronously copied into final storage and not retained. Only one
// matrix needs to be materialised at a time. The callback runs while the Vulkan
// lane is held and must not call Vulkan tools/operators or mutate prior results.
// Errors/cancellation return no usable partial owner; failed cleanup preserves
// an owner with the error for retrying Close. read is never retained.
func NewVkLinearQ5SetStream(ctx context.Context, shapes []VkLinearQ5Shape, read func(context.Context, int) ([]byte, error)) (result *VkLinearQ5Set, err error) {
	if read == nil {
		return nil, fmt.Errorf("Q5 stream: nil reader")
	}
	if err = vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	views, total, err := describeQ5Stream(ctx, shapes, vkLimits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	kernel, err := vkKernelCreateLocked(spirv_linear_q5_grouped_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	owner := &VkLinearQ5Set{kernel: kernel, views: views, bytes: total}
	committed := false
	defer func() {
		if !committed {
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
		var raw []byte
		raw, err = read(ctx, i)
		if err != nil {
			return nil, err
		}
		if len(raw) != v.outDim*(v.inDim/32)*22 {
			return nil, fmt.Errorf("Q5 stream: matrix%d payload", i)
		}
		dst := unsafe.Slice((*uint32)(unsafe.Add(owner.storage.mapped, uintptr(v.offset))), int(v.size/4))
		if err = packQ5Into(ctx, raw, dst); err != nil {
			return nil, err
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	runtime.KeepAlive(owner.storage)
	committed = true
	return owner, nil
}
