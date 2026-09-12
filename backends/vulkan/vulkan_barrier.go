package vulkan

import "unsafe"

// Classic core Vulkan barriers; no synchronization2 feature/extension required.
// Values/layout/signature checked against Vulkan-Headers v1.3.296 (see
// vulkan_abi.go). This package currently records compute-only commands on one
// queue, and allocates separate HOST_VISIBLE|HOST_COHERENT mapped buffers.
const (
	vkStructureTypeMemoryBarrier = 46
	vkPipelineStageComputeShader = 0x00000800
	vkPipelineStageHost          = 0x00004000
	vkAccessShaderRead           = 0x00000020
	vkAccessShaderWrite          = 0x00000040
	vkAccessHostRead             = 0x00002000
	vkAccessHostWrite            = 0x00004000
)

// VkMemoryBarrier: 24 bytes / alignment8 on the supported 64-bit FFI.
type vkMemoryBarrier struct {
	sType         uint32
	pNext         uintptr
	srcAccessMask uint32
	dstAccessMask uint32
}

var vkCmdPipelineBarrier func(VkCommandBuffer, uint32, uint32, uint32, uint32, unsafe.Pointer, uint32, unsafe.Pointer, uint32, unsafe.Pointer)

// vkComputeAcquireLocked must be recorded after Begin and before Dispatch.
// Queue submission already performs the domain operation for prior coherent
// host writes. Explicitly cover that path, plus shader writes from earlier
// commands on the SAME queue, including previous submissions. Execution scopes
// also order read-before-write reuse even though reads need no availability op.
// All buffers are treated as read/write until typed operator access is known.
func vkComputeAcquireLocked(cmd VkCommandBuffer) {
	barrier := vkMemoryBarrier{sType: vkStructureTypeMemoryBarrier,
		srcAccessMask: vkAccessHostWrite | vkAccessShaderWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite}
	vkCmdPipelineBarrier(cmd, vkPipelineStageHost|vkPipelineStageComputeShader,
		vkPipelineStageComputeShader, 0, 1, unsafe.Pointer(&barrier), 0, nil, 0, nil)
}

// vkComputeReleaseLocked must be recorded after Dispatch and before End.
// Shader writes become available to the host domain; HOST_COHERENT memory
// needs no vkInvalidateMappedMemoryRanges. A confirmed fence is STILL required
// before host reads or writes. Include both access types since a later checked
// upload may overwrite the allocation. Timeout/cancellation cannot bypass quarantine.
// This is not a transfer-stage, image, noncoherent or cross-queue barrier.
func vkComputeReleaseLocked(cmd VkCommandBuffer) {
	barrier := vkMemoryBarrier{sType: vkStructureTypeMemoryBarrier,
		srcAccessMask: vkAccessShaderWrite, dstAccessMask: vkAccessHostRead | vkAccessHostWrite}
	vkCmdPipelineBarrier(cmd, vkPipelineStageComputeShader, vkPipelineStageHost,
		0, 1, unsafe.Pointer(&barrier), 0, nil, 0, nil)
}
