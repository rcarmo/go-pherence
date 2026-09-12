package vulkan

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// This loader seam is private; tests inject only Go functions. All registration,
// rollback and publication run under vkLane. No native callback may reenter it.
type vkLoader struct {
	open  func() (uintptr, error)
	close func(uintptr) error
	bind  func(any, uintptr, string) error
}

func vkNativeLoader() vkLoader {
	return vkLoader{
		open: func() (uintptr, error) {
			h, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_LAZY)
			if err != nil {
				return purego.Dlopen("libvulkan.so", purego.RTLD_LAZY)
			}
			return h, err
		},
		close: purego.Dlclose,
		bind: func(target any, lib uintptr, name string) (err error) {
			defer func() {
				if p := recover(); p != nil {
					err = fmt.Errorf("bind %s: %v", name, p)
				}
			}()
			address, err := purego.Dlsym(lib, name)
			if err != nil {
				return err
			}
			if address == 0 {
				return fmt.Errorf("missing Vulkan symbol %s", name)
			}
			purego.RegisterFunc(target, address)
			return nil
		},
	}
}

type vkSymbol struct {
	target any
	name   string
}

type vkQueueFamilyProperties struct {
	queueFlags, queueCount, timestampValidBits uint32
	minImageTransferGranularity                [3]uint32
}

// vkInitLocked publishes state only after successful construction. No work has
// been submitted, so failure rollback may destroy the local pool/device/instance
// without waiting for idle. Failed create outputs are undefined and discarded.
// A failed dlclose keeps its handle as a bounded hold (no retry/reopen loop).
func vkInitLocked(loader vkLoader) bool {
	if vkLost || vkPending != nil || !vkNative64() {
		return false
	}
	if vkReady {
		return true
	}
	if vkLib != 0 {
		return false
	} // failed unload; process-level recovery only
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	lib, err := loader.open()
	if err != nil || lib == 0 {
		return false
	}
	var instance VkInstance
	var device VkDevice
	var pool VkCommandPool
	committed := false
	symbols := vkSymbols()
	clear := func() {
		for _, entry := range symbols {
			reflect.ValueOf(entry.target).Elem().SetZero()
		}
	}
	clear() // never allow a failed lookup to inherit a function from another loader
	defer func() {
		if committed {
			return
		}
		if pool != 0 {
			vkDestroyCommandPool(device, pool, nil)
		}
		if device != 0 {
			vkDestroyDevice(device, nil)
		}
		if instance != 0 {
			vkDestroyInstance(instance, nil)
		}
		clear() // no callable pointers into the library after unloading
		if err := loader.close(lib); err != nil {
			vkLib = lib
			debugf("[vulkan] library unload failed: %v\n", err)
		}
	}()
	// Resolve ALL mandatory creation/dispatch/destruction symbols before making
	// any object. Missing cleanup entry points therefore cannot create leaks.
	for _, entry := range symbols {
		if err := loader.bind(entry.target, lib, entry.name); err != nil || reflect.ValueOf(entry.target).Elem().IsNil() {
			return false
		}
	}
	return vkCreateInitialStateLocked(lib, &instance, &device, &pool, &committed)
}

// Native pointer type used in the symbol inventory is checked by RegisterFunc.
var (
	vkDestroyInstance    func(VkInstance, unsafe.Pointer)
	vkDestroyDevice      func(VkDevice, unsafe.Pointer)
	vkDestroyCommandPool func(VkDevice, VkCommandPool, unsafe.Pointer)
)

func vkSymbols() []vkSymbol {
	return []vkSymbol{
		{&vkCreateInstance, "vkCreateInstance"},
		{&vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices"},
		{&vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties"},
		{&vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties"},
		{&vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties"},
		{&vkCreateDevice, "vkCreateDevice"},
		{&vkGetDeviceQueue, "vkGetDeviceQueue"},
		{&vkCreateCommandPool, "vkCreateCommandPool"},
		{&vkCreateBuffer, "vkCreateBuffer"},
		{&vkAllocateMemory, "vkAllocateMemory"},
		{&vkBindBufferMemory, "vkBindBufferMemory"},
		{&vkMapMemory, "vkMapMemory"},
		{&vkUnmapMemory, "vkUnmapMemory"},
		{&vkCreateShaderModule, "vkCreateShaderModule"},
		{&vkCreateComputePipelines, "vkCreateComputePipelines"},
		{&vkCreatePipelineLayout, "vkCreatePipelineLayout"},
		{&vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout"},
		{&vkCreateDescriptorPool, "vkCreateDescriptorPool"},
		{&vkAllocateDescriptorSets, "vkAllocateDescriptorSets"},
		{&vkUpdateDescriptorSets, "vkUpdateDescriptorSets"},
		{&vkAllocateCommandBuffers, "vkAllocateCommandBuffers"},
		{&vkResetCommandBuffer, "vkResetCommandBuffer"},
		{&vkBeginCommandBuffer, "vkBeginCommandBuffer"},
		{&vkEndCommandBuffer, "vkEndCommandBuffer"},
		{&vkCmdBindPipeline, "vkCmdBindPipeline"},
		{&vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets"},
		{&vkCmdDispatch, "vkCmdDispatch"},
		{&vkCmdPipelineBarrier, "vkCmdPipelineBarrier"},
		{&vkQueueSubmit, "vkQueueSubmit"},
		{&vkQueueWaitIdle, "vkQueueWaitIdle"},
		{&vkCreateFence, "vkCreateFence"},
		{&vkWaitForFences, "vkWaitForFences"},
		{&vkResetFences, "vkResetFences"},
		{&vkCmdPushConstants, "vkCmdPushConstants"},
		{&vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements"},
		{&vkDestroyBuffer, "vkDestroyBuffer"},
		{&vkFreeMemory, "vkFreeMemory"},
		{&vkDestroyShaderModule, "vkDestroyShaderModule"},
		{&vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout"},
		{&vkDestroyPipelineLayout, "vkDestroyPipelineLayout"},
		{&vkDestroyPipeline, "vkDestroyPipeline"},
		{&vkDestroyDescriptorPool, "vkDestroyDescriptorPool"},
		{&vkFreeCommandBuffers, "vkFreeCommandBuffers"},
		{&vkDestroyFence, "vkDestroyFence"},
		{&vkDestroyInstance, "vkDestroyInstance"},
		{&vkDestroyDevice, "vkDestroyDevice"},
		{&vkDestroyCommandPool, "vkDestroyCommandPool"},
	}
}

func vkCreateInitialStateLocked(lib uintptr, instance *VkInstance, device *VkDevice, pool *VkCommandPool, committed *bool) bool {
	var physical VkPhysicalDevice
	var queue VkQueue
	var family uint32
	var deviceName string
	// Create instance (no extensions needed for compute-only)
	appInfo := struct {
		sType              uint32
		pNext              uintptr
		pApplicationName   *byte
		applicationVersion uint32
		pEngineName        *byte
		engineVersion      uint32
		apiVersion         uint32
	}{
		sType:      0,                     // VK_STRUCTURE_TYPE_APPLICATION_INFO
		apiVersion: (1 << 22) | (3 << 12), // Vulkan 1.3
	}

	createInfo := struct {
		sType                   uint32
		pNext                   uintptr
		flags                   uint32
		pApplicationInfo        unsafe.Pointer
		enabledLayerCount       uint32
		ppEnabledLayerNames     uintptr
		enabledExtensionCount   uint32
		ppEnabledExtensionNames uintptr
	}{
		sType:            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		pApplicationInfo: unsafe.Pointer(&appInfo),
	}

	if r := vkCreateInstance(unsafe.Pointer(&createInfo), nil, instance); r != VK_SUCCESS {
		*instance = 0
		debugf("[vulkan] vkCreateInstance failed: %d\n", r)
		return false
	}

	if *instance == 0 {
		return false
	}

	// Enumerate physical devices
	var devCount uint32
	if r := vkEnumeratePhysicalDevices(*instance, &devCount, nil); r != VK_SUCCESS {
		return false
	}
	if devCount == 0 || devCount > 64 {
		debugln("[vulkan] no physical devices found")
		return false
	}

	devs := make([]VkPhysicalDevice, devCount)
	if r := vkEnumeratePhysicalDevices(*instance, &devCount, &devs[0]); r != VK_SUCCESS {
		return false
	}
	if devCount == 0 || devCount > uint32(len(devs)) {
		return false
	}

	// Pick best compute device. CPU/software Vulkan implementations (e.g. llvmpipe)
	// are not an inference backend and are rejected by default; allow them only for
	// explicit shader debugging with GO_PHERENCE_VULKAN_ALLOW_CPU=1.

	allowCPU := os.Getenv("GO_PHERENCE_VULKAN_ALLOW_CPU") == "1"
	bestIdx := -1
	bestPriority := uint32(999)
	bestName := ""
	var bestLimits VulkanDeviceLimits
	for i := uint32(0); i < devCount; i++ {
		if devs[i] == 0 {
			return false
		}
		var props vkDeviceProperties
		vkGetPhysicalDeviceProperties(devs[i], unsafe.Pointer(&props))
		limits, err := vkLimitsFromProperties(props)
		if err != nil {
			continue
		} // skip unsupported/incomplete devices before preference
		name := string(props.deviceName[:])
		for j, b := range props.deviceName {
			if b == 0 {
				name = string(props.deviceName[:j])
				break
			}
		}

		priority := uint32(999)
		switch props.deviceType {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			priority = 0
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			priority = 1
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			priority = 2
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			if allowCPU {
				priority = 10
			}
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			if allowCPU {
				priority = 11
			}
		}
		if priority < bestPriority {
			bestIdx = int(i)
			bestPriority = priority
			bestName = name
			bestLimits = limits
		}
	}
	if bestIdx < 0 {
		debugln("[vulkan] no non-CPU Vulkan GPU found (set GO_PHERENCE_VULKAN_ALLOW_CPU=1 to allow software/CPU drivers)")
		return false
	}
	physical = devs[bestIdx]
	deviceName = bestName

	// Find compute queue family
	var queueCount uint32
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, nil)

	if queueCount == 0 || queueCount > 256 {
		return false
	}
	queueFams := make([]vkQueueFamilyProperties, queueCount)
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, unsafe.Pointer(&queueFams[0]))
	if queueCount == 0 || queueCount > uint32(len(queueFams)) {
		return false
	}

	found := false
	for i := uint32(0); i < queueCount; i++ {
		if queueFams[i].queueFlags&VK_QUEUE_COMPUTE_BIT != 0 && queueFams[i].queueCount > 0 {
			family = i
			found = true
			break
		}
	}
	if !found {
		debugln("[vulkan] no compute queue family found")
		return false
	}

	// Create logical device with one compute queue
	queuePriority := float32(1.0)
	queueCreateInfo := struct {
		sType            uint32
		pNext            uintptr
		flags            uint32
		queueFamilyIndex uint32
		queueCount       uint32
		pQueuePriorities unsafe.Pointer
	}{
		sType:            0x02, // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
		queueFamilyIndex: family,
		queueCount:       1,
		pQueuePriorities: unsafe.Pointer(&queuePriority),
	}

	deviceCreateInfo := struct {
		sType                   uint32
		pNext                   uintptr
		flags                   uint32
		queueCreateInfoCount    uint32
		pQueueCreateInfos       unsafe.Pointer
		enabledLayerCount       uint32
		ppEnabledLayerNames     uintptr
		enabledExtensionCount   uint32
		ppEnabledExtensionNames uintptr
		pEnabledFeatures        uintptr
	}{
		sType:                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		queueCreateInfoCount: 1,
		pQueueCreateInfos:    unsafe.Pointer(&queueCreateInfo),
	}

	if r := vkCreateDevice(physical, unsafe.Pointer(&deviceCreateInfo), nil, device); r != VK_SUCCESS {
		*device = 0
		debugf("[vulkan] vkCreateDevice failed: %d\n", r)
		return false
	}

	if *device == 0 {
		return false
	}
	vkGetDeviceQueue(*device, family, 0, &queue)
	if queue == 0 {
		return false
	}

	// Create command pool
	poolInfo := struct {
		sType            uint32
		pNext            uintptr
		flags            uint32
		queueFamilyIndex uint32
	}{
		sType:            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		flags:            0x02, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
		queueFamilyIndex: family,
	}

	if r := vkCreateCommandPool(*device, unsafe.Pointer(&poolInfo), nil, pool); r != VK_SUCCESS {
		*pool = 0
		debugf("[vulkan] vkCreateCommandPool failed: %d\n", r)
		return false
	}

	if *pool == 0 {
		return false
	}
	vkPublishInitialState(lib, *instance, physical, *device, queue, *pool, family, deviceName, bestLimits)
	*committed = true
	return true
}

func vkPublishInitialState(lib uintptr, instance VkInstance, physical VkPhysicalDevice, device VkDevice, queue VkQueue, pool VkCommandPool, family uint32, name string, limits VulkanDeviceLimits) {
	vkLib = lib
	vkInstance = instance
	vkPhysDev = physical
	vkDevice = device
	vkQueue = queue
	vkCmdPool = pool
	vkComputeQueueFamily = family
	vkDevName = name
	vkLimits = limits
	vkReady = true
}
