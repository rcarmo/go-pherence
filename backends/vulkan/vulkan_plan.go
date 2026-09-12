package vulkan

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"
)

// VkF32Stage describes one explicitly ordered dispatch. PushWords are raw32bit
// push data (use math.Float32bits for floats) and must exactly match the kernel's
// push range. Plan construction copies slices/view metadata; kernels and arena
// allocations remain caller-owned and are checked again before every run.
type VkF32Stage struct {
	Kernel    *VkComputeKernel
	Groups    [3]uint32
	Tensors   []*VkTensorF32
	PushWords []uint32
}

// VkF32Plan owns descriptor sets (one per stage), one command buffer and fence.
// It records/executes1..64 stages serially per Run, one submission/wait. No DAG,
// reordering, automatic tensor dtype/shape inference, uploads or model ownership.
// Copies share state and Close; underlying kernels/arenas are NOT closed by it.
type VkF32Plan struct{ state *vkF32PlanState }
type vkF32PlanStage struct {
	kernel   *VkComputeKernel
	groups   [3]uint32
	bindings []vkBufferBinding
	push     []uint32
	set      VkDescriptorSet
}
type vkF32PlanState struct {
	device      VkDevice
	queue       VkQueue
	pool        VkCommandPool
	descriptors VkDescriptorPool
	command     VkCommandBuffer
	fence       VkFence
	stages      []vkF32PlanStage
	closed      bool
}

func (s *vkF32PlanStage) pushPointer() unsafe.Pointer {
	if len(s.push) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.push[0])
}

func NewVkF32Plan(ctx context.Context, stages []VkF32Stage) (*VkF32Plan, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return nil, err
	}
	if len(stages) < 1 || len(stages) > 64 {
		return nil, fmt.Errorf("Vulkan F32 plan stages must be1..64")
	}
	p := &vkF32PlanState{device: vkDevice, queue: vkQueue, pool: vkCmdPool, stages: make([]vkF32PlanStage, len(stages))}
	var descriptorCount uint32
	for i, in := range stages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if in.Kernel == nil || in.Kernel.descSetLayout == 0 || len(in.Tensors) < 1 || len(in.Tensors) > 16 || len(in.PushWords) > 32 {
			return nil, fmt.Errorf("invalid Vulkan plan stage%d", i)
		}
		if len(in.PushWords)*4 != in.Kernel.pushSize {
			return nil, fmt.Errorf("Vulkan plan stage%d push length", i)
		}
		s := &p.stages[i]
		s.kernel = in.Kernel
		s.groups = in.Groups
		s.push = append([]uint32(nil), in.PushWords...)
		s.bindings = make([]vkBufferBinding, len(in.Tensors))
		for j, t := range in.Tensors {
			b, err := t.bindingLocked()
			if err != nil {
				return nil, err
			}
			s.bindings[j] = b
		}
		if err := s.kernel.validateBindingsLocked(s.groups[0], s.groups[1], s.groups[2], s.bindings, s.pushPointer()); err != nil {
			return nil, err
		}
		descriptorCount += uint32(len(s.bindings))
	}
	if vkCreateDescriptorPool == nil || vkAllocateDescriptorSets == nil || vkAllocateCommandBuffers == nil || vkResetCommandBuffer == nil || vkCreateFence == nil || vkDestroyDescriptorPool == nil || vkFreeCommandBuffers == nil || vkDestroyFence == nil {
		return nil, fmt.Errorf("Vulkan plan construction/cleanup functions unavailable")
	}
	committed := false
	defer func() {
		if !committed {
			p.destroyLocked()
		}
	}()
	size := struct{ kind, count uint32 }{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount}
	poolInfo := struct {
		sType                         uint32
		pNext                         uintptr
		flags, maxSets, poolSizeCount uint32
		pPoolSizes                    unsafe.Pointer
	}{sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, maxSets: uint32(len(stages)), poolSizeCount: 1, pPoolSizes: unsafe.Pointer(&size)}
	if r := vkCreateDescriptorPool(p.device, unsafe.Pointer(&poolInfo), nil, &p.descriptors); r != VK_SUCCESS {
		p.descriptors = 0
		return nil, fmt.Errorf("vkCreateDescriptorPool(plan): %d", r)
	}
	if p.descriptors == 0 {
		return nil, fmt.Errorf("null Vulkan plan descriptor pool")
	}
	layouts := make([]VkDescriptorSetLayout, len(stages))
	sets := make([]VkDescriptorSet, len(stages))
	for i, s := range p.stages {
		layouts[i] = s.kernel.descSetLayout
		if layouts[i] == 0 {
			return nil, fmt.Errorf("missing kernel descriptor layout")
		}
	}
	alloc := struct {
		sType   uint32
		pNext   uintptr
		pool    VkDescriptorPool
		count   uint32
		layouts unsafe.Pointer
	}{sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pool: p.descriptors, count: uint32(len(stages)), layouts: unsafe.Pointer(&layouts[0])}
	if r := vkAllocateDescriptorSets(p.device, unsafe.Pointer(&alloc), &sets[0]); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkAllocateDescriptorSets(plan): %d", r)
	}
	for i, set := range sets {
		if set == 0 {
			return nil, fmt.Errorf("null Vulkan plan descriptor set")
		}
		p.stages[i].set = set
	}
	cmdAlloc := struct {
		sType        uint32
		pNext        uintptr
		pool         VkCommandPool
		level, count uint32
	}{sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pool: p.pool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, count: 1}
	if r := vkAllocateCommandBuffers(p.device, unsafe.Pointer(&cmdAlloc), &p.command); r != VK_SUCCESS {
		p.command = 0
		return nil, fmt.Errorf("vkAllocateCommandBuffers(plan): %d", r)
	}
	if p.command == 0 {
		return nil, fmt.Errorf("null Vulkan plan command buffer")
	}
	fenceInfo := struct {
		sType uint32
		pNext uintptr
		flags uint32
	}{sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}
	if r := vkCreateFence(p.device, unsafe.Pointer(&fenceInfo), nil, &p.fence); r != VK_SUCCESS {
		p.fence = 0
		return nil, fmt.Errorf("vkCreateFence(plan): %d", r)
	}
	if p.fence == 0 {
		return nil, fmt.Errorf("null Vulkan plan fence")
	}
	runtime.KeepAlive(layouts)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	committed = true
	return &VkF32Plan{state: p}, nil
}

func (p *VkF32Plan) Run(ctx context.Context) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if p == nil || p.state == nil {
		return fmt.Errorf("nil Vulkan F32 plan")
	}
	s := p.state
	if s.closed {
		return ErrVulkanClosed
	}
	if s.device != vkDevice || s.queue != vkQueue || s.pool != vkCmdPool {
		return fmt.Errorf("Vulkan plan owner mismatch")
	}
	// Whole-plan preflight before ANY descriptor writes/recording. Owner closure
	// after construction must be rejected even for the final stage.
	pending := &vkPendingSubmission{plan: s}
	for _, stage := range s.stages {
		if err := stage.kernel.validateBindingsLocked(stage.groups[0], stage.groups[1], stage.groups[2], stage.bindings, stage.pushPointer()); err != nil {
			return err
		}
		for _, b := range stage.bindings {
			found := false
			for _, retained := range pending.buffers {
				if retained == b.buffer {
					found = true
					break
				}
			}
			if !found {
				pending.buffers = append(pending.buffers, b.buffer)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Explicit reset makes reuse valid after both successful execution and a
	// prior recording failure. Reset precedes descriptor mutation/recording.
	if r := vkResetCommandBuffer(s.command, 0); r != VK_SUCCESS {
		return vkDriverError("vkResetCommandBuffer(plan)", r)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Each stage's set is independent, even for repeated use of the same kernel.
	for _, stage := range s.stages {
		stage.kernel.updateBindingsLocked(stage.set, stage.bindings)
	}
	begin := vkCommandBufferBeginInfo{sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}
	if r := vkBeginCommandBuffer(s.command, unsafe.Pointer(&begin)); r != VK_SUCCESS {
		return vkDriverError("vkBeginCommandBuffer(plan)", r)
	}
	// Always finish recording after mid-record cancellation. Incomplete commands
	// are never submitted; successful End permits a later clean implicit reset.
	var cancelErr error
	vkComputeAcquireLocked(s.command)
	for i, stage := range s.stages {
		if cancelErr = ctx.Err(); cancelErr != nil {
			break
		}
		if i != 0 {
			vkComputeBetweenLocked(s.command)
		}
		vkCmdBindPipeline(s.command, VK_PIPELINE_BIND_POINT_COMPUTE, stage.kernel.pipeline)
		vkCmdBindDescriptorSets(s.command, VK_PIPELINE_BIND_POINT_COMPUTE, stage.kernel.pipelineLayout, 0, 1, &stage.set, 0, nil)
		if len(stage.push) > 0 {
			vkCmdPushConstants(s.command, stage.kernel.pipelineLayout, 0x20, 0, uint32(len(stage.push)*4), stage.pushPointer())
		}
		vkCmdDispatch(s.command, stage.groups[0], stage.groups[1], stage.groups[2])
	}
	vkComputeReleaseLocked(s.command)
	if r := vkEndCommandBuffer(s.command); r != VK_SUCCESS {
		return vkDriverError("vkEndCommandBuffer(plan)", r)
	}
	if cancelErr != nil {
		return cancelErr
	}
	err := vkSubmitLocked(ctx, s.device, s.queue, s.command, s.fence, pending)
	runtime.KeepAlive(s)
	return err
}

func (p *VkF32Plan) Close() error {
	if p == nil || p.state == nil {
		return nil
	}
	_ = vkAcquire(context.Background())
	defer vkRelease()
	s := p.state
	if s.closed {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if vkPending != nil && vkPending.plan == s {
		return ErrVulkanInFlight
	}
	if s.device != vkDevice || s.pool != vkCmdPool {
		return fmt.Errorf("Vulkan plan owner mismatch")
	}
	if vkDestroyFence == nil || vkFreeCommandBuffers == nil || vkDestroyDescriptorPool == nil {
		return fmt.Errorf("Vulkan plan cleanup functions unavailable")
	}
	s.destroyLocked()
	s.closed = true
	s.stages = nil
	return nil
}
func (s *vkF32PlanState) destroyLocked() {
	if s.fence != 0 {
		vkDestroyFence(s.device, s.fence, nil)
		s.fence = 0
	}
	if s.command != 0 {
		vkFreeCommandBuffers(s.device, s.pool, 1, &s.command)
		s.command = 0
	}
	if s.descriptors != 0 {
		vkDestroyDescriptorPool(s.device, s.descriptors, nil)
		s.descriptors = 0
	}
}
