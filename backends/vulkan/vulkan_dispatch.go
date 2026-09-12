package vulkan

// Vulkan compute dispatch: command buffers, descriptor binding, shader execution.
//
// Single-operation pipeline and shared admission/submit helpers for VkF32Plan.
// One serialized host lane retains at most one unresolved submission. Each
// operation or plan records broad dependencies; model residency is unfinished.
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
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// VkComputeKernel is a ready-to-dispatch Vulkan compute shader.
type VkComputeKernel struct {
	device         VkDevice
	queue          VkQueue
	commandPool    VkCommandPool
	closed         bool
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
// (multiple of4). Partial construction rolls back resources; Close releases a
// successful completed kernel. Resource objects must not be copied. These bounds
// also obey queried descriptor/push/local-size/shared-memory limits. The shader
// must fit InspectVulkanShader's conservative core envelope; this is not full
// SPIR-V validation; descriptor/push reflection is restricted to flat32-bit
// layouts. VulkanInit must already
// have completed without concurrent device/function-pointer replacement.
func VkKernelCreate(spirv []byte, numBuffers int, pushConstantSize int) (*VkComputeKernel, error) {
	if err := vkAcquire(context.Background()); err != nil {
		return nil, err
	}
	defer vkRelease()
	return vkKernelCreateLocked(spirv, numBuffers, pushConstantSize)
}

// Caller owns vkLane, including the one-shot cache construction below.
func vkKernelCreateLocked(spirv []byte, numBuffers int, pushConstantSize int) (*VkComputeKernel, error) {
	if err := vkStatusLocked(); err != nil {
		return nil, err
	}
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

	if len(spirv) < 20 || len(spirv) > 1<<20 || binary.LittleEndian.Uint32(spirv) != 0x07230203 {
		return nil, fmt.Errorf("invalid SPIR-V header/bound")
	}
	if err := vkCheckPipelineLimitsLocked(numBuffers, pushConstantSize); err != nil {
		return nil, err
	}
	// Parse and pass the SAME aligned owned code, never re-read caller bytes
	// after admission. Concurrent caller mutation during the copy is forbidden.
	code := make([]uint32, len(spirv)/4)
	for i := range code {
		code[i] = binary.LittleEndian.Uint32(spirv[4*i:])
	}
	contract, err := vkInspectSPIRV(code)
	if err != nil {
		return nil, err
	}
	if err := vkCheckShaderLimits(contract, vkLimits); err != nil {
		return nil, err
	}
	if err := vkCheckShaderInterface(contract, numBuffers, pushConstantSize); err != nil {
		return nil, err
	}
	defer func() { runtime.KeepAlive(code) }()
	if !vkKernelFunctionsReady() {
		return nil, fmt.Errorf("Vulkan kernel construction/cleanup functions unavailable")
	}
	// All failure rollback uses captured owner handles. No resources from this
	// construction have been submitted. Successful kernels capture their owner
	// and expose Close only after fence completion.
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
		device: device, queue: vkQueue, commandPool: commandPool,
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

// Dispatch retains the legacy one-second budget, now including queue admission.
// Caller pushData must point to at least pushSize readable bytes until return.
func (k *VkComputeKernel) Dispatch(groupsX, groupsY, groupsZ uint32, bufs []*VkBuf, pushData unsafe.Pointer) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	return k.DispatchContext(ctx, groupsX, groupsY, groupsZ, bufs, pushData)
}

// DispatchContext serializes descriptors, command-pool recording, queue and
// mapped accesses. Fence waits poll in <=10ms slices with a one-second cap;
// context cancellation prevents submission at checked boundaries. Once submitted,
// cancellation/timeout retains kernel+buffers and requires VulkanDrain. A driver
// call in progress cannot be interrupted. Push data remains caller-owned/read-only.
func (k *VkComputeKernel) DispatchContext(ctx context.Context, groupsX, groupsY, groupsZ uint32, bufs []*VkBuf, pushData unsafe.Pointer) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if len(bufs) > 16 {
		return fmt.Errorf("Vulkan binding count exceeds16")
	}
	bindings := make([]vkBufferBinding, len(bufs))
	for i, b := range bufs {
		bindings[i].buffer = b
		if b != nil {
			bindings[i].size = b.size
		}
	}
	return k.dispatchBindingsLocked(ctx, groupsX, groupsY, groupsZ, bindings, pushData)
}

type vkBufferBinding struct {
	buffer       *VkBuf
	offset, size uint64
}

// Caller owns the lane. Buffer owners (not views) enter pending retention.
func (k *VkComputeKernel) dispatchBindingsLocked(ctx context.Context, groupsX, groupsY, groupsZ uint32, bindings []vkBufferBinding, pushData unsafe.Pointer) error {
	if err := k.validateBindingsLocked(groupsX, groupsY, groupsZ, bindings, pushData); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Explicit reset makes reuse valid after both successful execution and a
	// prior recording failure. Reset precedes descriptor mutation/recording.
	if r := vkResetCommandBuffer(k.cmdBuf, 0); r != VK_SUCCESS {
		return vkDriverError("vkResetCommandBuffer", r)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	k.updateBindingsLocked(k.descSet, bindings)
	// Record command buffer
	beginInfo := vkCommandBufferBeginInfo{sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}
	if r := vkBeginCommandBuffer(k.cmdBuf, unsafe.Pointer(&beginInfo)); r != VK_SUCCESS {
		return vkDriverError("vkBeginCommandBuffer", r)
	}

	vkComputeAcquireLocked(k.cmdBuf)
	vkCmdBindPipeline(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline)
	vkCmdBindDescriptorSets(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &k.descSet, 0, nil)

	// Push constants if any
	if k.pushSize > 0 && pushData != nil {
		vkCmdPushConstants(k.cmdBuf, k.pipelineLayout, 0x20, 0, uint32(k.pushSize), pushData)
	}

	vkCmdDispatch(k.cmdBuf, groupsX, groupsY, groupsZ)
	vkComputeReleaseLocked(k.cmdBuf)

	if r := vkEndCommandBuffer(k.cmdBuf); r != VK_SUCCESS {
		return vkDriverError("vkEndCommandBuffer", r)
	}

	pending := &vkPendingSubmission{kernel: k}
	for _, binding := range bindings {
		buf := binding.buffer
		found := false
		for _, retained := range pending.buffers {
			if buf == retained {
				found = true
				break
			}
		}
		if !found {
			pending.buffers = append(pending.buffers, buf)
		}
	}
	err := vkSubmitLocked(ctx, k.device, k.queue, k.cmdBuf, k.fence, pending)
	runtime.KeepAlive(pushData)
	return err
}

func (k *VkComputeKernel) validateBindingsLocked(groupsX, groupsY, groupsZ uint32, bindings []vkBufferBinding, pushData unsafe.Pointer) error {
	if err := vkStatusLocked(); err != nil {
		return err
	}
	bufs := make([]*VkBuf, len(bindings))
	for i, binding := range bindings {
		bufs[i] = binding.buffer
	}
	if k != nil && k.closed {
		return ErrVulkanClosed
	}
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
	if vkUpdateDescriptorSets == nil || vkResetCommandBuffer == nil || vkBeginCommandBuffer == nil || vkCmdBindPipeline == nil || vkCmdBindDescriptorSets == nil || vkCmdDispatch == nil || vkCmdPipelineBarrier == nil || vkEndCommandBuffer == nil || vkResetFences == nil || vkQueueSubmit == nil || vkWaitForFences == nil || (k.pushSize > 0 && vkCmdPushConstants == nil) {
		return fmt.Errorf("Vulkan dispatch functions unavailable")
	}

	if k.device == 0 || k.queue == 0 || k.commandPool == 0 || k.device != vkDevice || k.queue != vkQueue || k.commandPool != vkCmdPool {
		return fmt.Errorf("Vulkan kernel owner mismatch")
	}
	for _, buf := range bufs {
		if buf.device != k.device {
			return fmt.Errorf("Vulkan buffer owner mismatch")
		}
		if buf.closed {
			return ErrVulkanClosed
		}
	}
	if err := vkCheckDispatchLimitsLocked(k, groupsX, groupsY, groupsZ, bufs); err != nil {
		return err
	}
	for _, b := range bindings {
		if b.size == 0 || b.offset >= b.buffer.size || b.size > b.buffer.size-b.offset || b.offset%vkLimits.StorageBufferOffsetAlignment != 0 {
			return fmt.Errorf("Vulkan descriptor range/alignment invalid")
		}
		if err := vkCheckBufferLimitLocked(b.size); err != nil {
			return err
		}
	}

	return nil
}

func (k *VkComputeKernel) updateBindingsLocked(set VkDescriptorSet, bindings []vkBufferBinding) {
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
	writes := make([]writeDS, len(bindings))
	bufInfos := make([]bufInfo, len(bindings))
	for i, binding := range bindings {
		buf := binding.buffer
		bufInfos[i] = bufInfo{buffer: buf.buf, offset: bindings[i].offset, rng: bindings[i].size} // explicit checked range
		writes[i] = writeDS{
			sType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			dstSet:          set,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			pBufferInfo:     unsafe.Pointer(&bufInfos[i]),
		}
	}
	vkUpdateDescriptorSets(k.device, uint32(len(writes)), unsafe.Pointer(&writes[0]), 0, nil)

}

func vkSubmitLocked(ctx context.Context, device VkDevice, queue VkQueue, cmd VkCommandBuffer, fence VkFence, pending *vkPendingSubmission) error {
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
		pCommandBuffers:    unsafe.Pointer(&cmd),
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if r := vkResetFences(device, 1, &fence); r != VK_SUCCESS {
		return vkDriverError("vkResetFences", r)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if r := vkQueueSubmit(queue, 1, unsafe.Pointer(&submitInfo), fence); r != VK_SUCCESS {
		// Conservative quarantine: failed submit/device loss is not proof that
		// native resources can safely be reused. Never poll an unsignalled fence
		// from a submission whose acceptance is unknown.
		pending.uncertain = true
		vkPending = pending
		return errors.Join(ErrVulkanInFlight, ErrVulkanUncertain, vkDriverError("vkQueueSubmit", r))
	}
	vkPending = pending
	return vkWaitPendingLocked(ctx, time.Second)
}

func vkKernelFunctionsReady() bool {
	return vkCreateShaderModule != nil && vkCreateDescriptorSetLayout != nil && vkCreatePipelineLayout != nil && vkCreateComputePipelines != nil && vkCreateDescriptorPool != nil && vkAllocateDescriptorSets != nil && vkAllocateCommandBuffers != nil && vkResetCommandBuffer != nil && vkCreateFence != nil && vkDestroyShaderModule != nil && vkDestroyDescriptorSetLayout != nil && vkDestroyPipelineLayout != nil && vkDestroyPipeline != nil && vkDestroyDescriptorPool != nil && vkFreeCommandBuffers != nil && vkDestroyFence != nil
}

// vkCmdPushConstants — needs to be registered
var vkCmdPushConstants func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer)
