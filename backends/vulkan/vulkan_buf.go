package vulkan

// Vulkan compute buffer and shader management.
//
// VkBuf: host-visible/coherent allocation (no separate staging allocation)
// VkShader: compiled SPIR-V compute pipeline

import (
	"context"
	"fmt"
	"unsafe"
)

// VkBuf is a host-visible/coherent buffer. Device-local memory is not required.
// Package APIs serialize host access and retain buffers during pending work.
// Resource objects must not be copied; callers cannot access the mapped pointer.
type VkBuf struct {
	device       VkDevice
	closed       bool
	deferredFree bool
	buf          VkBuffer
	mem          VkDeviceMemory
	size         uint64
	mapped       unsafe.Pointer
}

// VkBufAlloc allocates a Vulkan buffer accessible from both host and device.
func VkBufAlloc(sizeBytes int) (*VkBuf, error) {
	if err := vkAcquire(context.Background()); err != nil {
		return nil, err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return nil, err
	}
	if !vkNative64() {
		return nil, fmt.Errorf("Vulkan requires the current 64-bit FFI binding")
	}
	if !vkReady {
		return nil, fmt.Errorf("vulkan not initialized")
	}
	if sizeBytes <= 0 {
		return nil, fmt.Errorf("invalid vulkan buffer size=%d", sizeBytes)
	}
	// The current API binds the entire buffer as one storage descriptor.
	if err := vkCheckBufferLimitLocked(uint64(sizeBytes)); err != nil {
		return nil, err
	}

	if vkCreateBuffer == nil || vkGetBufferMemoryRequirements == nil || vkGetPhysicalDeviceMemoryProperties == nil || vkAllocateMemory == nil || vkBindBufferMemory == nil || vkMapMemory == nil || vkUnmapMemory == nil || vkDestroyBuffer == nil || vkFreeMemory == nil {
		return nil, fmt.Errorf("Vulkan buffer construction/cleanup functions unavailable")
	}
	device, physical := vkDevice, vkPhysDev
	var buf VkBuffer
	var mem VkDeviceMemory
	committed := false
	defer func() {
		if !committed {
			// A bound buffer must be destroyed before releasing its memory.
			if buf != 0 {
				vkDestroyBuffer(device, buf, nil)
			}
			if mem != 0 {
				vkFreeMemory(device, mem, nil)
			}
		}
	}()

	bufInfo := struct {
		sType                 uint32
		pNext                 uintptr
		flags                 uint32
		size                  uint64
		usage                 uint32
		sharingMode           uint32
		queueFamilyIndexCount uint32
		pQueueFamilyIndices   uintptr
	}{
		sType:       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		size:        uint64(sizeBytes),
		usage:       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		sharingMode: VK_SHARING_MODE_EXCLUSIVE,
	}

	if r := vkCreateBuffer(device, unsafe.Pointer(&bufInfo), nil, &buf); r != VK_SUCCESS {
		buf = 0 // output is undefined on failed creation
		return nil, fmt.Errorf("vkCreateBuffer: %d", r)
	}

	// Get memory requirements
	var reqs vkMemoryRequirements
	vkGetBufferMemoryRequirements(device, buf, unsafe.Pointer(&reqs))
	if reqs.size < uint64(sizeBytes) || reqs.alignment == 0 || reqs.alignment&(reqs.alignment-1) != 0 || reqs.memoryTypeBits == 0 {
		return nil, fmt.Errorf("invalid Vulkan memory requirements")
	}

	// Find host-visible + host-coherent memory type
	var props vkPhysicalDeviceMemoryProperties
	vkGetPhysicalDeviceMemoryProperties(physical, unsafe.Pointer(&props))
	if props.memoryTypeCount == 0 || props.memoryTypeCount > 32 || props.memoryHeapCount == 0 || props.memoryHeapCount > 16 {
		return nil, fmt.Errorf("invalid Vulkan memory property counts")
	}

	memTypeIdx := uint32(0xFFFFFFFF)
	wantFlags := uint32(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	for i := uint32(0); i < props.memoryTypeCount; i++ {
		if props.memoryTypes[i].heapIndex >= props.memoryHeapCount {
			return nil, fmt.Errorf("invalid Vulkan memory heap index")
		}
		if reqs.memoryTypeBits&(1<<i) != 0 && props.memoryTypes[i].propertyFlags&wantFlags == wantFlags && props.memoryHeaps[props.memoryTypes[i].heapIndex].size >= reqs.size {
			memTypeIdx = i
			break
		}
	}
	if memTypeIdx == 0xFFFFFFFF {
		return nil, fmt.Errorf("no suitable memory type")
	}

	allocInfo := struct {
		sType           uint32
		pNext           uintptr
		allocationSize  uint64
		memoryTypeIndex uint32
	}{
		sType:           VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		allocationSize:  reqs.size,
		memoryTypeIndex: memTypeIdx,
	}

	if r := vkAllocateMemory(device, unsafe.Pointer(&allocInfo), nil, &mem); r != VK_SUCCESS {
		mem = 0
		return nil, fmt.Errorf("vkAllocateMemory: %d", r)
	}

	if r := vkBindBufferMemory(device, buf, mem, 0); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkBindBufferMemory: %d", r)
	}

	// Map memory
	var mapped unsafe.Pointer
	if r := vkMapMemory(device, mem, 0, uint64(sizeBytes), 0, &mapped); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkMapMemory: %d", r)
	}

	if mapped == nil {
		return nil, fmt.Errorf("vkMapMemory returned nil pointer")
	}
	committed = true
	return &VkBuf{device: device, buf: buf, mem: mem, size: uint64(sizeBytes), mapped: mapped}, nil
}

// Upload copies float32 data to the buffer. Invalid inputs are ignored for
// legacy compatibility; callers that need diagnostics should use UploadChecked.
func (b *VkBuf) Upload(data []float32) { _ = b.UploadChecked(data) }

// UploadChecked copies float32 data to the buffer and reports malformed input.
func (b *VkBuf) UploadChecked(data []float32) error {
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if b != nil {
		if b.closed {
			return ErrVulkanClosed
		}
		if err := vkQuarantineLocked(); err != nil {
			return err
		}
		if vkUsesBufferLocked(b) {
			return ErrVulkanInFlight
		}
	}
	if len(data) == 0 {
		return nil
	}
	if b == nil || b.mapped == nil {
		return fmt.Errorf("vulkan buffer is not mapped")
	}
	bytes, ok := vkCheckedMulInt(len(data), 4)
	if !ok || uint64(bytes) > b.size {
		return fmt.Errorf("vulkan upload size=%d exceeds buffer size=%d", bytes, b.size)
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	dst := unsafe.Slice((*byte)(b.mapped), bytes)
	copy(dst, src)
	return nil
}

// Download copies float32 data from the buffer. Invalid inputs are ignored for
// legacy compatibility; callers that need diagnostics should use DownloadChecked.
func (b *VkBuf) Download(data []float32) { _ = b.DownloadChecked(data) }

// DownloadChecked copies float32 data from the buffer and reports malformed input.
func (b *VkBuf) DownloadChecked(data []float32) error {
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if b != nil {
		if b.closed {
			return ErrVulkanClosed
		}
		if err := vkQuarantineLocked(); err != nil {
			return err
		}
		if vkUsesBufferLocked(b) {
			return ErrVulkanInFlight
		}
	}
	if len(data) == 0 {
		return nil
	}
	if b == nil || b.mapped == nil {
		return fmt.Errorf("vulkan buffer is not mapped")
	}
	bytes, ok := vkCheckedMulInt(len(data), 4)
	if !ok || uint64(bytes) > b.size {
		return fmt.Errorf("vulkan download size=%d exceeds buffer size=%d", bytes, b.size)
	}
	src := unsafe.Slice((*byte)(b.mapped), bytes)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	copy(dst, src)
	return nil
}

// Free keeps the legacy signature. Pending buffers are retained and marked
// for deferred destruction after VulkanDrain confirms completion. After device
// loss/uncertain submission they stay retained until process teardown. Call
// FreeChecked when the caller needs to know that immediate destruction failed.
func (b *VkBuf) Free() {
	if b == nil {
		return
	}
	_ = vkAcquire(context.Background())
	defer vkRelease()
	if vkUsesBufferLocked(b) {
		b.deferredFree = true
		return
	}
	_ = b.freeLocked()
}

// FreeChecked never defers implicitly. An in-flight/lost error leaves the
// object untouched; drain then retry. Repeated successful frees are no-ops.
func (b *VkBuf) FreeChecked() error {
	if b == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if b.closed {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if vkUsesBufferLocked(b) {
		return ErrVulkanInFlight
	}
	return b.freeLocked()
}
func (b *VkBuf) freeLocked() error {
	if b.closed {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if b.device == 0 || b.device != vkDevice {
		return fmt.Errorf("Vulkan buffer owner mismatch")
	}
	if vkUnmapMemory == nil || vkDestroyBuffer == nil || vkFreeMemory == nil {
		return fmt.Errorf("Vulkan buffer cleanup functions unavailable")
	}
	if b.mapped != nil {
		vkUnmapMemory(b.device, b.mem)
		b.mapped = nil
	}
	if b.buf != 0 {
		vkDestroyBuffer(b.device, b.buf, nil)
		b.buf = 0
	}
	if b.mem != 0 {
		vkFreeMemory(b.device, b.mem, nil)
		b.mem = 0
	}
	b.closed = true
	b.deferredFree = false
	return nil
}
