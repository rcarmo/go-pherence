package vulkan

// Vulkan compute backend for inference on any GPU.
//
// Uses purego to dlopen libvulkan.so (no CGo required).
// Loads pre-compiled SPIR-V compute shaders for:
//   - GEMV (quantized and float)
//   - RMSNorm
//   - Vector operations (add, mul, scale, silu)
//   - Attention
//
// Architecture:
//   1. Instance → PhysicalDevice → LogicalDevice → Queue
//   2. Allocate device-local memory for weights
//   3. Create compute pipelines from SPIR-V modules
//   4. Record command buffers with dispatch commands
//   5. Submit and fence-wait for results
//
// This is the portable GPU path — works on:
//   - NVIDIA (desktop + Jetson)
//   - Intel (iGPU + Arc)
//   - AMD (RDNA)
//   - Qualcomm Adreno (Android/Linux)
//   - Apple (via MoltenVK)
//   - ARM Mali

import (
	"context"
	"unsafe"
)

// Vulkan types
type VkInstance uintptr
type VkPhysicalDevice uintptr
type VkDevice uintptr
type VkQueue uintptr
type VkCommandPool uintptr
type VkCommandBuffer uintptr
type VkBuffer uintptr
type VkDeviceMemory uintptr
type VkPipeline uintptr
type VkPipelineLayout uintptr
type VkShaderModule uintptr
type VkDescriptorSetLayout uintptr
type VkDescriptorPool uintptr
type VkDescriptorSet uintptr
type VkFence uintptr
type VkResult int32

const (
	VK_SUCCESS                                          VkResult = 0
	VK_TIMEOUT                                          VkResult = 2
	VK_ERROR_DEVICE_LOST                                VkResult = -4
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO                       = 1
	VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO                         = 3
	VK_STRUCTURE_TYPE_SUBMIT_INFO                                = 4
	VK_STRUCTURE_TYPE_FENCE_CREATE_INFO                          = 8
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO                         = 12
	VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO                  = 16
	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO               = 29
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO                = 30
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO          = 32
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO               = 34
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET                       = 35
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO                = 33
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO                   = 39
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO               = 40
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO                  = 42
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO                       = 5

	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT          = 0x00000020
	VK_BUFFER_USAGE_TRANSFER_SRC_BIT            = 0x00000001
	VK_BUFFER_USAGE_TRANSFER_DST_BIT            = 0x00000002
	VK_SHARING_MODE_EXCLUSIVE                   = 0
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT         = 0x00000001
	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT         = 0x00000002
	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT        = 0x00000004
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER           = 7
	VK_PIPELINE_BIND_POINT_COMPUTE              = 1
	VK_COMMAND_BUFFER_LEVEL_PRIMARY             = 0
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
	VK_QUEUE_COMPUTE_BIT                        = 0x00000002
	VK_PHYSICAL_DEVICE_TYPE_OTHER               = 0
	VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU      = 1
	VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU        = 2
	VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU         = 3
	VK_PHYSICAL_DEVICE_TYPE_CPU                 = 4

	VK_NULL_HANDLE = 0
)

// Vulkan state
var (
	vkLib                uintptr
	vkInstance           VkInstance
	vkPhysDev            VkPhysicalDevice
	vkDevice             VkDevice
	vkQueue              VkQueue
	vkCmdPool            VkCommandPool
	vkComputeQueueFamily uint32
	vkReady              bool
	vkDevName            string
)

// Vulkan function pointers
var (
	vkCreateInstance                         func(unsafe.Pointer, unsafe.Pointer, *VkInstance) VkResult
	vkEnumeratePhysicalDevices               func(VkInstance, *uint32, *VkPhysicalDevice) VkResult
	vkGetPhysicalDeviceProperties            func(VkPhysicalDevice, unsafe.Pointer)
	vkGetPhysicalDeviceMemoryProperties      func(VkPhysicalDevice, unsafe.Pointer)
	vkGetPhysicalDeviceQueueFamilyProperties func(VkPhysicalDevice, *uint32, unsafe.Pointer)
	vkCreateDevice                           func(VkPhysicalDevice, unsafe.Pointer, unsafe.Pointer, *VkDevice) VkResult
	vkGetDeviceQueue                         func(VkDevice, uint32, uint32, *VkQueue)
	vkCreateCommandPool                      func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkCommandPool) VkResult
	vkCreateBuffer                           func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkBuffer) VkResult
	vkAllocateMemory                         func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDeviceMemory) VkResult
	vkBindBufferMemory                       func(VkDevice, VkBuffer, VkDeviceMemory, uint64) VkResult
	vkMapMemory                              func(VkDevice, VkDeviceMemory, uint64, uint64, uint32, *unsafe.Pointer) VkResult
	vkUnmapMemory                            func(VkDevice, VkDeviceMemory)
	vkCreateShaderModule                     func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult
	vkCreateComputePipelines                 func(VkDevice, uintptr, uint32, unsafe.Pointer, unsafe.Pointer, *VkPipeline) VkResult
	vkCreatePipelineLayout                   func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkPipelineLayout) VkResult
	vkCreateDescriptorSetLayout              func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorSetLayout) VkResult
	vkCreateDescriptorPool                   func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorPool) VkResult
	vkAllocateDescriptorSets                 func(VkDevice, unsafe.Pointer, *VkDescriptorSet) VkResult
	vkUpdateDescriptorSets                   func(VkDevice, uint32, unsafe.Pointer, uint32, unsafe.Pointer)
	vkAllocateCommandBuffers                 func(VkDevice, unsafe.Pointer, *VkCommandBuffer) VkResult
	vkBeginCommandBuffer                     func(VkCommandBuffer, unsafe.Pointer) VkResult
	vkEndCommandBuffer                       func(VkCommandBuffer) VkResult
	vkCmdBindPipeline                        func(VkCommandBuffer, uint32, VkPipeline)
	vkCmdBindDescriptorSets                  func(VkCommandBuffer, uint32, VkPipelineLayout, uint32, uint32, *VkDescriptorSet, uint32, unsafe.Pointer)
	vkCmdDispatch                            func(VkCommandBuffer, uint32, uint32, uint32)
	vkQueueSubmit                            func(VkQueue, uint32, unsafe.Pointer, VkFence) VkResult
	vkQueueWaitIdle                          func(VkQueue) VkResult
	vkCreateFence                            func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkFence) VkResult
	vkWaitForFences                          func(VkDevice, uint32, *VkFence, uint32, uint64) VkResult
	vkResetFences                            func(VkDevice, uint32, *VkFence) VkResult
	vkGetBufferMemoryRequirements            func(VkDevice, VkBuffer, unsafe.Pointer)
	vkDestroyBuffer                          func(VkDevice, VkBuffer, unsafe.Pointer)
	vkFreeMemory                             func(VkDevice, VkDeviceMemory, unsafe.Pointer)
	vkDestroyShaderModule                    func(VkDevice, VkShaderModule, unsafe.Pointer)
	vkDestroyDescriptorSetLayout             func(VkDevice, VkDescriptorSetLayout, unsafe.Pointer)
	vkDestroyPipelineLayout                  func(VkDevice, VkPipelineLayout, unsafe.Pointer)
	vkDestroyPipeline                        func(VkDevice, VkPipeline, unsafe.Pointer)
	vkDestroyDescriptorPool                  func(VkDevice, VkDescriptorPool, unsafe.Pointer)
	vkFreeCommandBuffers                     func(VkDevice, VkCommandPool, uint32, *VkCommandBuffer)
	vkDestroyFence                           func(VkDevice, VkFence, unsafe.Pointer)
)

// VulkanInit initializes the Vulkan compute backend.
func VulkanInit() bool {
	if err := vkAcquire(context.Background()); err != nil {
		return false
	}
	defer vkRelease()
	return vkInitLocked(vkNativeLoader())
}

// VulkanReady reports current submission admission, not just initialisation.
// Retained or quarantined work makes it false until a successful drain.
func VulkanReady() bool {
	if err := vkAcquire(context.Background()); err != nil {
		return false
	}
	defer vkRelease()
	return vkReady && vkStatusLocked() == nil
}

// VulkanDeviceName returns the Vulkan device name.
func VulkanDeviceName() string {
	if err := vkAcquire(context.Background()); err != nil {
		return ""
	}
	defer vkRelease()
	return vkDevName
}
