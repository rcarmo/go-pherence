package vulkan

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// ABI pinned to Vulkan-Headers vulkan-sdk-1.3.296.0. Only packed 4x8
// integer-dot is enabled; Int8 and 8-bit storage are not requested.
type vkIntegerDotFeatures struct {
	sType   uint32
	pNext   unsafe.Pointer
	enabled uint32
}
type vkFeatures2 struct {
	sType    uint32
	pNext    unsafe.Pointer
	features [55]uint32
}
type vkIntegerDotProperties struct {
	sType       uint32
	pNext       unsafe.Pointer
	accelerated [30]uint32
}
type vkProperties2 struct {
	sType      uint32
	pNext      unsafe.Pointer
	properties vkDeviceProperties
}

var vkGetPhysicalDeviceFeatures2 func(VkPhysicalDevice, unsafe.Pointer)
var vkGetPhysicalDeviceProperties2 func(VkPhysicalDevice, unsafe.Pointer)
var vkIntegerDotEnabled bool

// VulkanInitIntegerDot creates an explicitly enabled Vulkan 1.3 integer-dot
// device. Requires accelerated unsigned weights × signed packed activations.
// It never recreates an existing baseline device or silently falls back.
func VulkanInitIntegerDot() bool {
	if err := vkAcquire(context.Background()); err != nil {
		return false
	}
	defer vkRelease()
	return vkInitLockedMode(vkNativeLoader(), true)
}
func VulkanIntegerDotEnabled() bool {
	if err := vkAcquire(context.Background()); err != nil {
		return false
	}
	defer vkRelease()
	return vkReady && vkStatusLocked() == nil && vkIntegerDotEnabled
}
func vkQueryIntegerDot(physical VkPhysicalDevice) *vkIntegerDotFeatures {
	if vkGetPhysicalDeviceFeatures2 == nil || vkGetPhysicalDeviceProperties2 == nil {
		return nil
	}
	feature := &vkIntegerDotFeatures{sType: 1000280000}
	props := &vkIntegerDotProperties{sType: 1000280001}
	var pin runtime.Pinner
	pin.Pin(feature)
	pin.Pin(props)
	defer pin.Unpin()
	query := vkFeatures2{sType: 1000059000, pNext: unsafe.Pointer(feature)}
	vkGetPhysicalDeviceFeatures2(physical, unsafe.Pointer(&query))
	pq := vkProperties2{sType: 1000059001, pNext: unsafe.Pointer(props)}
	vkGetPhysicalDeviceProperties2(physical, unsafe.Pointer(&pq))
	runtime.KeepAlive(feature)
	runtime.KeepAlive(props)
	runtime.KeepAlive(query)
	runtime.KeepAlive(pq)
	// index5: integerDotProduct4x8BitPackedMixedSignednessAccelerated.
	if feature.enabled != 1 || props.accelerated[5] != 1 {
		fmt.Fprintf(os.Stderr, "vulkan: integer-dot refused: feature=%d packed unsigned/signed/mixed acceleration=%v\n", feature.enabled, props.accelerated[3:6])
		return nil
	}
	return &vkIntegerDotFeatures{sType: 1000280000, enabled: 1}
}
