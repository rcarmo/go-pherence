package vulkan

// Vulkan compute dispatch: command buffers, descriptor binding, shader execution.
//
// Legacy single-operation pipeline; not a model-ready queue owner. Shared
// descriptors/queue and timeout handling still need lifetime synchronisation.
//   VkComputeKernel: compiled shader + pipeline + descriptor layout
//   Dispatch: record command buffer → bind descriptors → submit → bounded wait
//
// The pattern for each operation:
//   1. Bind pipeline
//   2. Update descriptor set with buffer bindings
//   3. Push constants (dimensions, eps, etc.)
//   4. Dispatch workgroups
//   5. Submit + fence wait

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"unsafe"
)

// VkComputeKernel is a ready-to-dispatch Vulkan compute shader.
type VkComputeKernel struct {
	pipeline       VkPipeline
	pipelineLayout VkPipelineLayout
	descSetLayout  VkDescriptorSetLayout
	descPool       VkDescriptorPool
	descSet        VkDescriptorSet
	cmdBuf         VkCommandBuffer
	fence          VkFence
	numBuffers     int
	pushSize       int
}

// VkKernelCreate builds a compute kernel with1..16 buffers and0..128 push bytes
// (multiple of4). Partial construction rolls back resources; successful kernels
// remain owned by the legacy caller with no automatic destruction. These bounds
// are host admission only: device/shader feature and descriptor limits are not
// negotiated here. VulkanInit must already have completed without concurrent
// device/function-pointer replacement.
func VkKernelCreate(spirv []byte, numBuffers int, pushConstantSize int) (*VkComputeKernel, error) {
	if !vkNative64() {
		return nil, fmt.Errorf("Vulkan requires the current 64-bit FFI binding")
	}
	if !vkReady {
		return nil, fmt.Errorf("vulkan not initialized")
	}
	if len(spirv) == 0 || len(spirv)%4 != 0 {
		return nil, fmt.Errorf("invalid SPIR-V length=%d", len(spirv))
	}
	if numBuffers <= 0 || numBuffers > 16 {
		return nil, fmt.Errorf("invalid Vulkan descriptor buffer count=%d", numBuffers)
	}
	if pushConstantSize < 0 || pushConstantSize > 128 || pushConstantSize%4 != 0 {
		return nil, fmt.Errorf("invalid Vulkan push constant size=%d", pushConstantSize)
	}

	if len(spirv) < 20 || len(spirv) > 16<<20 || binary.LittleEndian.Uint32(spirv) != 0x07230203 {
		return nil, fmt.Errorf("invalid SPIR-V header/bound")
	}
	if !vkKernelFunctionsReady() {
		return nil, fmt.Errorf("Vulkan kernel construction/cleanup functions unavailable")
	}
	// All failure rollback uses captured owner handles. No resources from this
	// construction have been submitted. Successful kernel destruction still
	// needs a separate fence/queue ownership API, not blind deferred teardown.
	device, commandPool := vkDevice, vkCmdPool
	var shaderModule VkShaderModule
	var descSetLayout VkDescriptorSetLayout
	var pipelineLayout VkPipelineLayout
	var pipeline VkPipeline
	var descPool VkDescriptorPool
	var cmdBuf VkCommandBuffer
	var fence VkFence
	committed := false
	defer func() {
		if !committed {
			if fence != 0 {
				vkDestroyFence(device, fence, nil)
			}
			if cmdBuf != 0 {
				vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf)
			}
			if descPool != 0 {
				vkDestroyDescriptorPool(device, descPool, nil)
			}
			if pipeline != 0 {
				vkDestroyPipeline(device, pipeline, nil)
			}
			if pipelineLayout != 0 {
				vkDestroyPipelineLayout(device, pipelineLayout, nil)
			}
			if descSetLayout != 0 {
				vkDestroyDescriptorSetLayout(device, descSetLayout, nil)
			}
		}
		if shaderModule != 0 {
			vkDestroyShaderModule(device, shaderModule, nil)
		}
	}()
	// Use aligned uint32 SPIR-V storage even for an unaligned caller byte slice.
	code := make([]uint32, len(spirv)/4)
	for i := range code {
		code[i] = binary.LittleEndian.Uint32(spirv[4*i:])
	}
	defer func() { runtime.KeepAlive(code) }()

	// Create shader module
	moduleInfo := struct {
		sType    uint32
		pNext    uintptr
		flags    uint32
		codeSize uint64
		pCode    unsafe.Pointer
	}{
		sType:    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		codeSize: uint64(len(spirv)),
		pCode:    unsafe.Pointer(&code[0]),
	}
	if r := vkCreateShaderModule(device, unsafe.Pointer(&moduleInfo), nil, &shaderModule); r != VK_SUCCESS {
		shaderModule = 0 // non-pipeline create output is undefined on failure
		return nil, fmt.Errorf("vkCreateShaderModule: %d", r)
	}

	// Descriptor set layout: N storage buffers
	type descBinding struct {
		binding, descType, descCount, stageFlags uint32
		pSamplers                                uintptr
	}
	bindings := make([]descBinding, numBuffers)
	for i := range bindings {
		bindings[i] = descBinding{
			binding:    uint32(i),
			descType:   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			descCount:  1,
			stageFlags: 0x20, // COMPUTE
		}
	}
	layoutInfo := struct {
		sType        uint32
		pNext        uintptr
		flags        uint32
		bindingCount uint32
		pBindings    unsafe.Pointer
	}{
		sType:        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount: uint32(numBuffers),
		pBindings:    unsafe.Pointer(&bindings[0]),
	}
	if r := vkCreateDescriptorSetLayout(device, unsafe.Pointer(&layoutInfo), nil, &descSetLayout); r != VK_SUCCESS {
		descSetLayout = 0
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", r)
	}

	// Push constant range
	pushRange := struct {
		stageFlags uint32
		offset     uint32
		size       uint32
	}{stageFlags: 0x20, size: uint32(pushConstantSize)}

	// Pipeline layout with push constants
	plInfo := struct {
		sType                  uint32
		pNext                  uintptr
		flags                  uint32
		setLayoutCount         uint32
		pSetLayouts            unsafe.Pointer
		pushConstantRangeCount uint32
		_                      uint32
		pPushConstantRanges    unsafe.Pointer
	}{
		sType:                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount:         1,
		pSetLayouts:            unsafe.Pointer(&descSetLayout),
		pushConstantRangeCount: 1,
		pPushConstantRanges:    unsafe.Pointer(&pushRange),
	}
	if pushConstantSize == 0 {
		plInfo.pushConstantRangeCount = 0
		plInfo.pPushConstantRanges = nil
	}
	if r := vkCreatePipelineLayout(device, unsafe.Pointer(&plInfo), nil, &pipelineLayout); r != VK_SUCCESS {
		pipelineLayout = 0
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", r)
	}

	// Compute pipeline
	entryName := append([]byte("main"), 0)
	type stageCI struct {
		sType  uint32
		pNext  uintptr
		flags  uint32
		stage  uint32
		module VkShaderModule
		pName  unsafe.Pointer
		pSpec  uintptr
	}
	stage := stageCI{
		sType:  0x12, // PIPELINE_SHADER_STAGE_CREATE_INFO
		stage:  0x20, // COMPUTE
		module: shaderModule,
		pName:  unsafe.Pointer(&entryName[0]),
	}
	type computePCI struct {
		sType  uint32
		pNext  uintptr
		flags  uint32
		_      uint32
		stage  stageCI
		layout VkPipelineLayout
		basePH uintptr
		basePI int32
	}
	pci := computePCI{
		sType:  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		stage:  stage,
		layout: pipelineLayout,
	}
	if r := vkCreateComputePipelines(device, 0, 1, unsafe.Pointer(&pci), nil, &pipeline); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", r)
	}

	// Descriptor pool
	poolSize := struct {
		descType  uint32
		descCount uint32
	}{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32(numBuffers)}
	poolInfo := struct {
		sType         uint32
		pNext         uintptr
		flags         uint32
		maxSets       uint32
		poolSizeCount uint32
		_             uint32
		pPoolSizes    unsafe.Pointer
	}{
		sType:         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		maxSets:       1,
		poolSizeCount: 1,
		pPoolSizes:    unsafe.Pointer(&poolSize),
	}
	if r := vkCreateDescriptorPool(device, unsafe.Pointer(&poolInfo), nil, &descPool); r != VK_SUCCESS {
		descPool = 0
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", r)
	}

	// Allocate descriptor set
	allocInfo := struct {
		sType          uint32
		pNext          uintptr
		descriptorPool VkDescriptorPool
		descSetCount   uint32
		_              uint32
		pSetLayouts    unsafe.Pointer
	}{
		sType:          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool: descPool,
		descSetCount:   1,
		pSetLayouts:    unsafe.Pointer(&descSetLayout),
	}
	var descSet VkDescriptorSet
	if r := vkAllocateDescriptorSets(device, unsafe.Pointer(&allocInfo), &descSet); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", r)
	}

	// Allocate command buffer
	cmdAllocInfo := struct {
		sType           uint32
		pNext           uintptr
		commandPool     VkCommandPool
		level           uint32
		commandBufCount uint32
	}{
		sType:           VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool:     commandPool,
		level:           VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		commandBufCount: 1,
	}
	if r := vkAllocateCommandBuffers(device, unsafe.Pointer(&cmdAllocInfo), &cmdBuf); r != VK_SUCCESS {
		cmdBuf = 0
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", r)
	}

	// Create fence
	fenceInfo := struct {
		sType uint32
		pNext uintptr
		flags uint32
	}{sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}
	if r := vkCreateFence(device, unsafe.Pointer(&fenceInfo), nil, &fence); r != VK_SUCCESS {
		fence = 0
		return nil, fmt.Errorf("vkCreateFence: %d", r)
	}

	committed = true
	return &VkComputeKernel{
		pipeline:       pipeline,
		pipelineLayout: pipelineLayout,
		descSetLayout:  descSetLayout,
		descPool:       descPool,
		descSet:        descSet,
		cmdBuf:         cmdBuf,
		fence:          fence,
		numBuffers:     numBuffers,
		pushSize:       pushConstantSize,
	}, nil
}

// Dispatch executes one legacy synchronous operation. Calls sharing kernels,
// buffers or the global queue must be externally serialised. A timeout is NOT
// completion and does not make buffers safe to free/reuse. Caller pushData must
// point to at least pushSize readable bytes until recording returns.
func (k *VkComputeKernel) Dispatch(groupsX, groupsY, groupsZ uint32, bufs []*VkBuf, pushData unsafe.Pointer) error {
	if k == nil || k.pipeline == 0 || k.pipelineLayout == 0 || k.descSet == 0 || k.cmdBuf == 0 || k.fence == 0 {
		return fmt.Errorf("vulkan dispatch on uninitialized kernel")
	}
	if groupsX == 0 || groupsY == 0 || groupsZ == 0 {
		return fmt.Errorf("vulkan dispatch has zero workgroups (%d,%d,%d)", groupsX, groupsY, groupsZ)
	}
	if len(bufs) != k.numBuffers {
		return fmt.Errorf("vulkan dispatch buffer count=%d want=%d", len(bufs), k.numBuffers)
	}
	for i, buf := range bufs {
		if buf == nil || buf.buf == 0 || buf.mem == 0 || buf.size == 0 {
			return fmt.Errorf("vulkan dispatch buffer %d is not initialized", i)
		}
	}

	if !vkNative64() {
		return fmt.Errorf("Vulkan requires the current 64-bit FFI binding")
	}
	if k.pushSize < 0 || k.pushSize > 128 || k.pushSize%4 != 0 || (k.pushSize > 0 && pushData == nil) {
		return fmt.Errorf("missing/invalid Vulkan push constants")
	}
	if vkUpdateDescriptorSets == nil || vkBeginCommandBuffer == nil || vkCmdBindPipeline == nil || vkCmdBindDescriptorSets == nil || vkCmdDispatch == nil || vkEndCommandBuffer == nil || vkResetFences == nil || vkQueueSubmit == nil || vkWaitForFences == nil || (k.pushSize > 0 && vkCmdPushConstants == nil) {
		return fmt.Errorf("Vulkan dispatch functions unavailable")
	}

	// Update descriptor set with buffer bindings
	type bufInfo struct {
		buffer VkBuffer
		offset uint64
		rng    uint64 // VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF
	}
	type writeDS struct {
		sType            uint32
		pNext            uintptr
		dstSet           VkDescriptorSet
		dstBinding       uint32
		dstArrayElement  uint32
		descriptorCount  uint32
		descriptorType   uint32
		pImageInfo       uintptr
		pBufferInfo      unsafe.Pointer
		pTexelBufferView uintptr
	}
	writes := make([]writeDS, len(bufs))
	bufInfos := make([]bufInfo, len(bufs))
	for i, buf := range bufs {
		bufInfos[i] = bufInfo{buffer: buf.buf, rng: 0xFFFFFFFFFFFFFFFF}
		writes[i] = writeDS{
			sType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			dstSet:          k.descSet,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			pBufferInfo:     unsafe.Pointer(&bufInfos[i]),
		}
	}
	vkUpdateDescriptorSets(vkDevice, uint32(len(writes)), unsafe.Pointer(&writes[0]), 0, nil)

	// Record command buffer
	beginInfo := vkCommandBufferBeginInfo{sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}
	if r := vkBeginCommandBuffer(k.cmdBuf, unsafe.Pointer(&beginInfo)); r != VK_SUCCESS {
		return fmt.Errorf("vkBeginCommandBuffer: %d", r)
	}

	vkCmdBindPipeline(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline)
	vkCmdBindDescriptorSets(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &k.descSet, 0, nil)

	// Push constants if any
	if k.pushSize > 0 && pushData != nil {
		vkCmdPushConstants(k.cmdBuf, k.pipelineLayout, 0x20, 0, uint32(k.pushSize), pushData)
	}

	vkCmdDispatch(k.cmdBuf, groupsX, groupsY, groupsZ)

	if r := vkEndCommandBuffer(k.cmdBuf); r != VK_SUCCESS {
		return fmt.Errorf("vkEndCommandBuffer: %d", r)
	}

	// Submit
	submitInfo := struct {
		sType                uint32
		pNext                uintptr
		waitSemaphoreCount   uint32
		_                    uint32
		pWaitSemaphores      uintptr
		pWaitDstStageMask    uintptr
		commandBufferCount   uint32
		_2                   uint32
		pCommandBuffers      unsafe.Pointer
		signalSemaphoreCount uint32
		_3                   uint32
		pSignalSemaphores    uintptr
	}{
		sType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
		commandBufferCount: 1,
		pCommandBuffers:    unsafe.Pointer(&k.cmdBuf),
	}
	if r := vkResetFences(vkDevice, 1, &k.fence); r != VK_SUCCESS {
		return fmt.Errorf("vkResetFences: %d", r)
	}
	if r := vkQueueSubmit(vkQueue, 1, unsafe.Pointer(&submitInfo), k.fence); r != VK_SUCCESS {
		return fmt.Errorf("vkQueueSubmit: %d", r)
	}

	// Wait for completion (1 second timeout)
	if r := vkWaitForFences(vkDevice, 1, &k.fence, 1, 1_000_000_000); r != VK_SUCCESS {
		return fmt.Errorf("vkWaitForFences: %d", r)
	}

	return nil
}

func vkKernelFunctionsReady() bool {
	return vkCreateShaderModule != nil && vkCreateDescriptorSetLayout != nil && vkCreatePipelineLayout != nil && vkCreateComputePipelines != nil && vkCreateDescriptorPool != nil && vkAllocateDescriptorSets != nil && vkAllocateCommandBuffers != nil && vkCreateFence != nil && vkDestroyShaderModule != nil && vkDestroyDescriptorSetLayout != nil && vkDestroyPipelineLayout != nil && vkDestroyPipeline != nil && vkDestroyDescriptorPool != nil && vkFreeCommandBuffers != nil && vkDestroyFence != nil
}

// vkCmdPushConstants — needs to be registered
var vkCmdPushConstants func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer)
