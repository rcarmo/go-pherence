package vulkan

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"
)

// VkTensorArena owns one host-visible/coherent VkBuf. F32 tensors are disjoint
// aligned ranges in that buffer. Allocation is monotonic, with no reset/reuse or
// per-tensor Free; Close releases the whole buffer only when idle. The capacity
// is currently <=maxStorageBufferRange, not an arbitrarily large device heap.
// It is not a resident model, an execution graph or a quantised tensor format.
type VkTensorArena struct{ state *vkTensorArenaState }
type vkTensorArenaState struct {
	buffer    *VkBuf
	used      uint64
	tensors   uint32
	alignment uint64
}

// VkTensorF32 is a contiguous row-major view with owned shape metadata. It keeps
// its arena state/backing buffer reachable and exposes no raw handle or mapped
// pointer. Copies share storage/closure, not ownership of a second allocation.
type VkTensorF32 struct {
	arena        *vkTensorArenaState
	offset, size uint64
	shape        [8]int
	rank         int
}

type VkTensorArenaStats struct {
	CapacityBytes, UsedBytes uint64
	Tensors                  uint32
	Closed, InFlight         bool
}

func NewVkTensorArena(ctx context.Context, capacityBytes int) (*VkTensorArena, error) {
	return newVkTensorArena(ctx, capacityBytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
}

// NewVkSharedTensorArena requires coherent, CPU-cached native memory. No
// uncached fallback is permitted. It enables scoped CPU/SIMD access to the
// same allocation used by Vulkan; it does not initialise a device.
func NewVkSharedTensorArena(ctx context.Context, capacityBytes int) (*VkTensorArena, error) {
	return newVkTensorArena(ctx, capacityBytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
}

func newVkTensorArena(ctx context.Context, capacityBytes int, flags uint32) (*VkTensorArena, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	buffer, err := vkBufAllocFlagsLocked(capacityBytes, flags)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		_ = buffer.freeLocked()
		return nil, err
	}
	return &VkTensorArena{state: &vkTensorArenaState{buffer: buffer, alignment: max(uint64(4), vkLimits.StorageBufferOffsetAlignment)}}, nil
}

// AllocF32 reserves a fresh, disjoint range. Contents are uninitialised until an
// Upload or a shader fully writes them. No native allocation occurs. Rank1..8,
// dimensions>0 and the complete shape/rounded offset must fit before mutation.
// Pending use of ANY view blocks reservations/transfers/Close for the arena.
func (a *VkTensorArena) AllocF32(ctx context.Context, shape ...int) (*VkTensorF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if a == nil {
		return nil, fmt.Errorf("nil Vulkan tensor arena")
	}
	if err := vkArenaReadyLocked(a.state); err != nil {
		return nil, err
	}
	bytes, err := vkF32ShapeBytes(shape)
	if err != nil {
		return nil, err
	}
	s := a.state
	if s.tensors >= 65536 {
		return nil, fmt.Errorf("Vulkan arena tensor count exceeds65536")
	}
	// Difference-based checks avoid overflowing used+alignment-1 or offset+size.
	pad := (s.alignment - s.used%s.alignment) % s.alignment
	if s.used > s.buffer.size || pad > s.buffer.size-s.used {
		return nil, fmt.Errorf("Vulkan arena exhausted by alignment")
	}
	offset := s.used + pad
	if bytes > s.buffer.size-offset {
		return nil, fmt.Errorf("Vulkan arena capacity exceeded")
	}
	t := &VkTensorF32{arena: s, offset: offset, size: bytes, rank: len(shape)}
	copy(t.shape[:], shape)
	s.used = offset + bytes
	s.tensors++
	return t, nil
}
func vkF32ShapeBytes(shape []int) (uint64, error) {
	if len(shape) < 1 || len(shape) > 8 {
		return 0, fmt.Errorf("Vulkan F32 rank must be1..8")
	}
	n := uint64(1)
	maxElements := uint64(int(^uint(0)>>1)) / 4
	for _, d := range shape {
		if d <= 0 || uint64(d) > maxElements/n {
			return 0, fmt.Errorf("Vulkan F32 shape overflow/zero dimension")
		}
		n *= uint64(d)
	}
	return n * 4, nil
}
func vkArenaReadyLocked(s *vkTensorArenaState) error {
	if s == nil || s.buffer == nil {
		return fmt.Errorf("uninitialized Vulkan tensor arena")
	}
	b := s.buffer
	if b.closed {
		return ErrVulkanClosed
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if vkUsesBufferLocked(b) {
		return ErrVulkanInFlight
	}
	if b.device == 0 || b.device != vkDevice || b.mapped == nil || b.buf == 0 || b.mem == 0 {
		return fmt.Errorf("Vulkan tensor arena owner/storage invalid")
	}
	return nil
}
func (a *VkTensorArena) Close() error {
	if a == nil || a.state == nil {
		return nil
	}
	// FreeChecked owns the lane; state/buffer references never change.
	return a.state.buffer.FreeChecked()
}
func (a *VkTensorArena) Stats() VkTensorArenaStats {
	_ = vkAcquire(context.Background())
	defer vkRelease()
	if a == nil || a.state == nil {
		return VkTensorArenaStats{Closed: true}
	}
	s := a.state
	return VkTensorArenaStats{CapacityBytes: s.buffer.size, UsedBytes: s.used, Tensors: s.tensors, Closed: s.buffer.closed, InFlight: vkUsesBufferLocked(s.buffer)}
}
func (t *VkTensorF32) Shape() []int {
	if t == nil || t.rank < 1 || t.rank > 8 {
		return nil
	}
	return append([]int(nil), t.shape[:t.rank]...)
}
func (t *VkTensorF32) Elements() int {
	if t == nil {
		return 0
	}
	return int(t.size / 4)
}

// PrefixRows returns a non-owning [rows,width] view over the first rows of a
// rank-two tensor. It does not allocate storage or alter the parent. The
// arena owner must outlive all uses of the view; transfers and stages still
// check closure and in-flight state against that owner.
func (t *VkTensorF32) PrefixRows(ctx context.Context, rows int) (*VkTensorF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if t == nil || t.rank != 2 || rows < 1 || rows > t.shape[0] {
		return nil, fmt.Errorf("invalid Vulkan F32 prefix rows")
	}
	if _, err := t.bindingLocked(); err != nil {
		return nil, err
	}
	view := *t
	view.shape[0] = rows
	view.size = uint64(rows) * uint64(t.shape[1]) * 4
	return &view, nil
}

// Upload/Download require the exact tensor length, so an undersized transfer
// cannot silently leave a tensor partially updated. Cancellation is checked at
// lane admission and before the copy; the single bounded copy is not preempted.
// Caller slices must not be accessed concurrently during transfer.
func (t *VkTensorF32) Upload(ctx context.Context, data []float32) error {
	return t.transfer(ctx, data, true)
}
func (t *VkTensorF32) Download(ctx context.Context, data []float32) error {
	return t.transfer(ctx, data, false)
}
func (t *VkTensorF32) bindingLocked() (vkBufferBinding, error) {
	if t == nil {
		return vkBufferBinding{}, fmt.Errorf("nil Vulkan tensor")
	}
	if err := vkArenaReadyLocked(t.arena); err != nil {
		return vkBufferBinding{}, err
	}
	if t.size == 0 || t.offset >= t.arena.buffer.size || t.size > t.arena.buffer.size-t.offset {
		return vkBufferBinding{}, fmt.Errorf("invalid Vulkan tensor range")
	}
	return vkBufferBinding{buffer: t.arena.buffer, offset: t.offset, size: t.size}, nil
}
func (t *VkTensorF32) transfer(ctx context.Context, data []float32, upload bool) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	binding, err := t.bindingLocked()
	if err != nil {
		return err
	}
	if uint64(len(data)) != t.size/4 {
		return fmt.Errorf("Vulkan tensor transfer length=%d want=%d", len(data), t.size/4)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	mapped := unsafe.Slice((*float32)(unsafe.Add(binding.buffer.mapped, uintptr(t.offset))), len(data))
	if upload {
		copy(mapped, data)
	} else {
		copy(data, mapped)
	}
	runtime.KeepAlive(binding.buffer)
	return nil
}

// WithCPURead borrows the tensor's native, CPU-cached mapping synchronously.
// The slice MUST NOT escape the callback, be mutated, or be used asynchronously.
// The callback must not reenter any Vulkan API (including Close/Stats): the
// package lane remains held, excluding submission, transfer and destruction.
// GPU-written data is accessible only after a confirmed fence and the existing
// compute-to-host barrier. An in-flight/uncertain/lost owner is rejected.
func (t *VkTensorF32) WithCPURead(ctx context.Context, consume func([]float32) error) error {
	return t.withCPU(ctx, consume)
}

// WithCPUWrite borrows the same mapping for a synchronous producer. The slice
// must not escape or be used asynchronously, and Vulkan APIs must not be called
// from the callback. Errors/cancellation/panics release the lease but do not
// roll back writes; callers must discard or fully overwrite failed contents
// before submitting them. Subsequent dispatch uses the host-to-compute barrier.
func (t *VkTensorF32) WithCPUWrite(ctx context.Context, produce func([]float32) error) error {
	return t.withCPU(ctx, produce)
}

func (t *VkTensorF32) withCPU(ctx context.Context, callback func([]float32) error) error {
	if callback == nil {
		return fmt.Errorf("nil Vulkan CPU callback")
	}
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	binding, err := t.bindingLocked()
	if err != nil {
		return err
	}
	const required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
	if binding.buffer.memoryFlags&required != required {
		return fmt.Errorf("Vulkan CPU borrow requires host-visible/coherent/cached memory")
	}
	if t.offset%4 != 0 || t.size%4 != 0 || uintptr(binding.buffer.mapped)%4 != 0 || t.size/4 > uint64(int(^uint(0)>>1)) {
		return fmt.Errorf("invalid Vulkan CPU F32 alignment/size")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	defer runtime.KeepAlive(binding.buffer)
	mapped := unsafe.Slice((*float32)(unsafe.Add(binding.buffer.mapped, uintptr(t.offset))), int(t.size/4))
	if err := callback(mapped); err != nil {
		return err
	}
	return ctx.Err()
}

// DispatchF32Context uses tensor byte ranges but retains their backing buffers,
// once each, through the existing synchronous lane/quarantine lifecycle. An
// arena referenced by one binding is blocked as a whole, even disjoint views.
// This method does not assert shader numeric types or model shape semantics:
// callers must choose an F32-compatible kernel and correctly sized push data.
func (k *VkComputeKernel) DispatchF32Context(ctx context.Context, x, y, z uint32, tensors []*VkTensorF32, pushData unsafe.Pointer) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if len(tensors) < 1 || len(tensors) > 16 {
		return fmt.Errorf("Vulkan tensor binding count must be1..16")
	}
	bindings := make([]vkBufferBinding, len(tensors))
	for i, t := range tensors {
		binding, err := t.bindingLocked()
		if err != nil {
			return err
		}
		bindings[i] = binding
	}
	return k.dispatchBindingsLocked(ctx, x, y, z, bindings, pushData)
}
