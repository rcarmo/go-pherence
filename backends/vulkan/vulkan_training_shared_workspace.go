package vulkan

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// VulkanSharedWorkspaceDriver is an explicit adapter for guarded, stable-G
// shared residency. Existing VulkanResidentDriver allocation/routes are unchanged.
// Exact copy currently uses coherent mapped native storage under the Vulkan lane;
// no F32 arithmetic, temporary host staging or shader fallback is involved.
// This CPU copy must be included in transfer-inclusive performance measurements.
type VulkanSharedWorkspaceDriver struct{ base *VulkanResidentDriver }

func NewVulkanSharedWorkspaceDriver(op *VkTrainingF32) *VulkanSharedWorkspaceDriver {
	return &VulkanSharedWorkspaceDriver{base: &VulkanResidentDriver{Operator: op}}
}

type vkSharedOwner struct {
	driver *VulkanSharedWorkspaceDriver
	raw    *vkResidentBuffer
}
type vkSharedPrefix struct {
	driver *VulkanSharedWorkspaceDriver
	parent *vkSharedOwner
	tensor *VkTensorF32
	access *train.SharedResidentAccess
}

var _ train.ResidentSharedWorkspaceDriver = (*VulkanSharedWorkspaceDriver)(nil)
var _ train.ResidentDeviceScaleDriver = (*VulkanSharedWorkspaceDriver)(nil)
var _ train.ResidentGlobalNormDriver = (*VulkanSharedWorkspaceDriver)(nil)
var _ train.ResidentCompactEmbeddingDriver = (*VulkanSharedWorkspaceDriver)(nil)
var _ train.ResidentHandoffTapeDriver = (*VulkanSharedWorkspaceDriver)(nil)

func (d *VulkanSharedWorkspaceDriver) ready(ctx context.Context) error {
	if d == nil || d.base == nil || d.base.Operator == nil || d.base.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/terminal shared Vulkan driver/context")
	}
	return ctx.Err()
}
func (b *vkSharedOwner) Upload(ctx context.Context, x []float32) error {
	if b == nil || b.driver == nil {
		return fmt.Errorf("nil shared owner")
	}
	v, e := b.driver.resolve(ctx, b, true)
	if e != nil {
		return e
	}
	return v.tensor.Upload(ctx, x)
}
func (b *vkSharedOwner) Download(ctx context.Context, x []float32) error {
	if b == nil || b.driver == nil {
		return fmt.Errorf("nil shared owner")
	}
	v, e := b.driver.resolve(ctx, b, false)
	if e != nil {
		return e
	}
	return v.tensor.Download(ctx, x)
}
func (b *vkSharedOwner) Close() error {
	if b == nil || b.driver == nil || b.driver.base == nil || b.raw == nil || b.raw.arena == nil {
		return fmt.Errorf("nil shared owner")
	}
	if b.driver.base.terminal != nil {
		return fmt.Errorf("terminal shared owner retained")
	}
	return b.raw.Close()
}
func (b *vkSharedPrefix) Upload(ctx context.Context, x []float32) error {
	if b == nil || b.driver == nil {
		return fmt.Errorf("nil shared prefix")
	}
	v, e := b.driver.resolve(ctx, b, true)
	if e != nil {
		return e
	}
	return v.tensor.Upload(ctx, x)
}
func (b *vkSharedPrefix) Download(ctx context.Context, x []float32) error {
	if b == nil || b.driver == nil {
		return fmt.Errorf("nil shared prefix")
	}
	v, e := b.driver.resolve(ctx, b, false)
	if e != nil {
		return e
	}
	return v.tensor.Download(ctx, x)
}
func (*vkSharedPrefix) Close() error { return fmt.Errorf("shared prefix cannot close parent") }

// Private resolution is synchronous; raw wrappers never leave the adapter or
// enter caller-visible tape roots. Guard tokens are rechecked on every operation.
func (d *VulkanSharedWorkspaceDriver) resolve(ctx context.Context, b train.ResidentBuffer, write bool) (*vkResidentBuffer, error) {
	if e := d.ready(ctx); e != nil {
		return nil, e
	}
	var raw *vkResidentBuffer
	switch v := b.(type) {
	case *vkSharedOwner:
		if v == nil || v.driver != d || v.raw == nil || v.raw.borrowed {
			return nil, fmt.Errorf("foreign shared owner")
		}
		raw = v.raw
	case *vkSharedPrefix:
		if v == nil || v.driver != d || v.parent == nil || v.parent.driver != d || v.parent.raw == nil || v.parent.raw.tensor == nil || v.tensor == nil {
			return nil, fmt.Errorf("foreign shared prefix")
		}
		if e := v.access.Validate(ctx, d, write); e != nil {
			return nil, e
		}
		if v.tensor.arena != v.parent.raw.tensor.arena || v.tensor.offset != v.parent.raw.tensor.offset || v.tensor.size > v.parent.raw.tensor.size {
			return nil, fmt.Errorf("shared prefix parent/range")
		}
		raw = &vkResidentBuffer{tensor: v.tensor, arena: v.parent.raw.arena, borrowed: true}
	case *vkResidentBuffer:
		// Only completed tape outputs may cross into shared accumulation as reads.
		if v == nil || write || !v.borrowed || v.sourceTape == nil || v.sourceTape.closed || v.sourceTape.terminal != nil || v.sourceTape.op != d.base.Operator {
			return nil, fmt.Errorf("foreign/expired shared source")
		}
		raw = v
	default:
		return nil, fmt.Errorf("foreign shared buffer")
	}
	if raw.tensor == nil {
		return nil, fmt.Errorf("nil shared tensor")
	}
	if e := vkAcquire(ctx); e != nil {
		return nil, e
	}
	defer vkRelease()
	if _, e := raw.tensor.bindingLocked(); e != nil {
		return nil, e
	}
	return raw, nil
}
func (d *VulkanSharedWorkspaceDriver) buffers(ctx context.Context, write []bool, values ...train.ResidentBuffer) ([]train.ResidentBuffer, error) {
	if len(write) != len(values) {
		return nil, fmt.Errorf("shared permission arity")
	}
	out := make([]train.ResidentBuffer, len(values))
	for i, b := range values {
		v, e := d.resolve(ctx, b, write[i])
		if e != nil {
			return nil, e
		}
		out[i] = v
	}
	return out, nil
}
func (d *VulkanSharedWorkspaceDriver) NewBuffer(ctx context.Context, n int) (train.ResidentBuffer, error) {
	if e := d.ready(ctx); e != nil {
		return nil, e
	}
	b, e := d.base.NewBuffer(ctx, n)
	if b == nil {
		return nil, e
	}
	raw, ok := b.(*vkResidentBuffer)
	if !ok {
		return nil, fmt.Errorf("shared allocator type")
	}
	return &vkSharedOwner{driver: d, raw: raw}, e
}
func (d *VulkanSharedWorkspaceDriver) ValidateSharedInventory(ctx context.Context, owners []train.ResidentBuffer, extents []int) error {
	if len(owners) == 0 || len(owners) != len(extents) || len(owners) > 5*4096+7 {
		return fmt.Errorf("shared inventory count")
	}
	raw := make([]*vkResidentBuffer, len(owners))
	for i, b := range owners {
		o, ok := b.(*vkSharedOwner)
		if !ok || o == nil || o.driver != d {
			return fmt.Errorf("shared inventory owner")
		}
		v, e := d.resolve(ctx, b, false)
		if e != nil {
			return e
		}
		if extents[i] < 1 || v.tensor.rank != 1 || v.tensor.Elements() != extents[i] {
			return fmt.Errorf("shared inventory extent")
		}
		raw[i] = v
	}
	if e := vkAcquire(ctx); e != nil {
		return e
	}
	defer vkRelease()
	for i, b := range raw {
		a, e := b.tensor.bindingLocked()
		if e != nil {
			return e
		}
		for j := 0; j < i; j++ {
			z, e := raw[j].tensor.bindingLocked()
			if e != nil {
				return e
			}
			if vkBindingsOverlap(a, z) {
				return fmt.Errorf("shared inventory physical alias")
			}
		}
	}
	return nil
}
func (d *VulkanSharedWorkspaceDriver) Prefix(ctx context.Context, parent train.ResidentBuffer, n int, access *train.SharedResidentAccess) (train.ResidentBuffer, error) {
	if e := access.Validate(ctx, d, false); e != nil {
		return nil, e
	}
	owner, ok := parent.(*vkSharedOwner)
	if !ok || owner == nil || owner.driver != d {
		return nil, fmt.Errorf("shared prefix owner")
	}
	raw, e := d.resolve(ctx, parent, false)
	if e != nil {
		return nil, e
	}
	if n < 1 || raw.tensor.rank != 1 || n > raw.tensor.Elements() {
		return nil, fmt.Errorf("shared prefix extent")
	}
	view := *raw.tensor
	view.size = uint64(n) * 4
	view.shape = [8]int{n}
	view.rank = 1
	return &vkSharedPrefix{driver: d, parent: owner, tensor: &view, access: access}, nil
}
func (d *VulkanSharedWorkspaceDriver) ValidateSharedAccumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	v, e := d.buffers(ctx, []bool{true, false, false}, dst, a, b)
	if e != nil {
		return e
	}
	if e = d.base.ValidateAccumulate(ctx, v[0], v[1], v[2], n, scale); e != nil {
		return e
	}
	// Enforce complete pairwise disjointness, including the two read inputs.
	return d.disjoint(ctx, v)
}
func (d *VulkanSharedWorkspaceDriver) ValidateAccumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	return d.ValidateSharedAccumulate(ctx, dst, a, b, n, scale)
}
func (d *VulkanSharedWorkspaceDriver) disjoint(ctx context.Context, v []train.ResidentBuffer) error {
	raw, e := vkResidentBuffers(v...)
	if e != nil {
		return e
	}
	if e = vkAcquire(ctx); e != nil {
		return e
	}
	defer vkRelease()
	for i, x := range raw {
		a, e := x.bindingLocked()
		if e != nil {
			return e
		}
		for j := 0; j < i; j++ {
			b, e := raw[j].bindingLocked()
			if e != nil {
				return e
			}
			if vkBindingsOverlap(a, b) {
				return fmt.Errorf("shared physical alias")
			}
		}
	}
	return nil
}
func (d *VulkanSharedWorkspaceDriver) Accumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	if e := d.ValidateSharedAccumulate(ctx, dst, a, b, n, scale); e != nil {
		return e
	}
	v, e := d.buffers(ctx, []bool{true, false, false}, dst, a, b)
	if e != nil {
		return e
	}
	return d.base.Accumulate(ctx, v[0], v[1], v[2], n, scale)
}
func (d *VulkanSharedWorkspaceDriver) ValidateCopy(ctx context.Context, dst, source train.ResidentBuffer, n int) error {
	if _, e := train.NormStatsShape(n); e != nil {
		return e
	}
	v, e := d.buffers(ctx, []bool{true, false}, dst, source)
	if e != nil {
		return e
	}
	for _, b := range v {
		if b.(*vkResidentBuffer).tensor.Elements() != n {
			return fmt.Errorf("shared copy extent")
		}
	}
	return d.disjoint(ctx, v)
}
func (d *VulkanSharedWorkspaceDriver) Copy(ctx context.Context, dst, source train.ResidentBuffer, n int) error {
	if e := d.ValidateCopy(ctx, dst, source, n); e != nil {
		return e
	}
	v, e := d.buffers(ctx, []bool{true, false}, dst, source)
	if e != nil {
		return e
	}
	if e = vkAcquire(ctx); e != nil {
		return e
	}
	defer vkRelease()
	out, e := v[0].(*vkResidentBuffer).tensor.bindingLocked()
	if e != nil {
		return e
	}
	in, e := v[1].(*vkResidentBuffer).tensor.bindingLocked()
	if e != nil {
		return e
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	// bindingLocked rejects pending/uncertain device owners. All prior compute
	// plans have completed a compute-to-host barrier/fence; coherence needs no flush.
	bytes := n * 4
	copy(unsafe.Slice((*byte)(unsafe.Add(out.buffer.mapped, uintptr(out.offset))), bytes), unsafe.Slice((*byte)(unsafe.Add(in.buffer.mapped, uintptr(in.offset))), bytes))
	runtime.KeepAlive(v)
	return ctx.Err()
}
func (d *VulkanSharedWorkspaceDriver) ValidateNormStats(ctx context.Context, dst, source train.ResidentBuffer, n int, positive bool) error {
	v, e := d.buffers(ctx, []bool{true, false}, dst, source)
	if e != nil {
		return e
	}
	raw, _ := vkResidentBuffers(v...)
	if raw[1].Elements() != n {
		return fmt.Errorf("shared norm extent")
	}
	_, e = d.base.Operator.NormStatsStage(ctx, raw[0], raw[1], positive)
	return e
}
func (d *VulkanSharedWorkspaceDriver) NormStats(ctx context.Context, dst, source train.ResidentBuffer, n int, positive bool) error {
	if e := d.ValidateNormStats(ctx, dst, source, n, positive); e != nil {
		return e
	}
	v, e := d.buffers(ctx, []bool{true, false}, dst, source)
	if e != nil {
		return e
	}
	return d.base.NormStats(ctx, v[0], v[1], n, positive)
}
func (d *VulkanSharedWorkspaceDriver) ValidateNormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	v, e := d.buffers(ctx, []bool{true, false, false}, dst, a, b)
	if e != nil {
		return e
	}
	return d.base.ValidateNormMerge(ctx, v[0], v[1], v[2], n, previous, mode)
}
func (d *VulkanSharedWorkspaceDriver) NormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	v, e := d.buffers(ctx, []bool{true, false, false}, dst, a, b)
	if e != nil {
		return e
	}
	return d.base.NormMerge(ctx, v[0], v[1], v[2], n, previous, mode)
}
func (d *VulkanSharedWorkspaceDriver) ValidateNormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	v, e := d.buffers(ctx, []bool{true, false}, dst, a)
	if e != nil {
		return e
	}
	return d.base.ValidateNormClip(ctx, v[0], v[1], elements, maxNorm)
}
func (d *VulkanSharedWorkspaceDriver) NormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	v, e := d.buffers(ctx, []bool{true, false}, dst, a)
	if e != nil {
		return e
	}
	return d.base.NormClip(ctx, v[0], v[1], elements, maxNorm)
}
func (d *VulkanSharedWorkspaceDriver) ValidateUpdate(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, s train.AdamWStep) error {
	x, e := d.buffers(ctx, []bool{true, false, true, true, true}, p, g, m, v, ema)
	if e != nil {
		return e
	}
	return d.base.ValidateUpdate(ctx, x[0], x[1], x[2], x[3], x[4], n, s)
}
func (d *VulkanSharedWorkspaceDriver) Update(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, s train.AdamWStep) error {
	x, e := d.buffers(ctx, []bool{true, false, true, true, true}, p, g, m, v, ema)
	if e != nil {
		return e
	}
	return d.base.Update(ctx, x[0], x[1], x[2], x[3], x[4], n, s)
}
func (d *VulkanSharedWorkspaceDriver) ValidateScaledUpdate(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, s train.AdamWStep) error {
	x, e := d.buffers(ctx, []bool{true, false, true, true, true, false}, p, g, m, v, ema, clip)
	if e != nil {
		return e
	}
	return d.base.ValidateScaledUpdate(ctx, x[0], x[1], x[2], x[3], x[4], x[5], n, s)
}
func (d *VulkanSharedWorkspaceDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, s train.AdamWStep) error {
	x, e := d.buffers(ctx, []bool{true, false, true, true, true, false}, p, g, m, v, ema, clip)
	if e != nil {
		return e
	}
	return d.base.UpdateScaled(ctx, x[0], x[1], x[2], x[3], x[4], x[5], n, s)
}
func (d *VulkanSharedWorkspaceDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocab int, scale float32) error {
	x, e := d.buffers(ctx, []bool{true, false, false, false}, out, previous, compact, index)
	if e != nil {
		return e
	}
	return d.base.ValidateCompactEmbeddingAccumulate(ctx, x[0], x[1], x[2], x[3], rows, width, vocab, scale)
}
func (d *VulkanSharedWorkspaceDriver) CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocab int, scale float32) error {
	x, e := d.buffers(ctx, []bool{true, false, false, false}, out, previous, compact, index)
	if e != nil {
		return e
	}
	return d.base.CompactEmbeddingAccumulate(ctx, x[0], x[1], x[2], x[3], rows, width, vocab, scale)
}
func (d *VulkanSharedWorkspaceDriver) tapeBindings(ctx context.Context, inputs map[int]train.ResidentBuffer) (map[int]train.ResidentBuffer, error) {
	out := map[int]train.ResidentBuffer{}
	for id, b := range inputs {
		if _, ok := b.(*vkSharedPrefix); ok {
			return nil, fmt.Errorf("shared prefix cannot enter retained tape")
		}
		v, e := d.resolve(ctx, b, false)
		if e != nil {
			return nil, e
		}
		out[id] = v
	}
	return out, nil
}
func (d *VulkanSharedWorkspaceDriver) ValidateHandoffTape(ctx context.Context, t train.Tape, inputs map[int]train.ResidentBuffer, maxBytes int64) error {
	v, e := d.tapeBindings(ctx, inputs)
	if e != nil {
		return e
	}
	return d.base.ValidateHandoffTape(ctx, t, v, maxBytes)
}
func (d *VulkanSharedWorkspaceDriver) NewBoundTape(ctx context.Context, t train.Tape, inputs map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	v, e := d.tapeBindings(ctx, inputs)
	if e != nil {
		return nil, e
	}
	return d.base.NewBoundTape(ctx, t, v, maxBytes)
}
func (d *VulkanSharedWorkspaceDriver) NewReuseBoundTape(ctx context.Context, t train.Tape, inputs map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	v, e := d.tapeBindings(ctx, inputs)
	if e != nil {
		return nil, e
	}
	return d.base.NewReuseBoundTape(ctx, t, v, maxBytes)
}
func (d *VulkanSharedWorkspaceDriver) NewZeroBoundTape(ctx context.Context, t train.Tape, inputs map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	v, e := d.tapeBindings(ctx, inputs)
	if e != nil {
		return nil, e
	}
	return d.base.NewZeroBoundTape(ctx, t, v, maxBytes)
}
func (d *VulkanSharedWorkspaceDriver) Close() error {
	if d == nil || d.base == nil {
		return nil
	}
	return d.base.Close()
}
