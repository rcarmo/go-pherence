package vulkan

import "unsafe"

// Explicit native layouts for the 64-bit ABI supported by the current loader.
// Vulkan-Headers v1.3.296 vulkan_core.h SHA256:
// 50af5a157c8aab7d90dcd929a05758b4dc3e78a619f46552bfd5cde67b2d46c1.
// Existing uintptr non-dispatchable handles are NOT a correct 32-bit binding.
// Init/allocation/kernel construction reject other pointer widths before FFI.
type vkCommandBufferBeginInfo struct {
	sType            uint32
	pNext            uintptr
	flags            uint32
	pInheritanceInfo unsafe.Pointer
}

type vkMemoryRequirements struct {
	size, alignment uint64
	memoryTypeBits  uint32
}
type vkMemoryType struct{ propertyFlags, heapIndex uint32 }
type vkMemoryHeap struct {
	size  uint64
	flags uint32
}
type vkPhysicalDeviceMemoryProperties struct {
	memoryTypeCount uint32
	memoryTypes     [32]vkMemoryType
	memoryHeapCount uint32
	memoryHeaps     [16]vkMemoryHeap
}

func vkNative64() bool { return unsafe.Sizeof(uintptr(0)) == 8 }
