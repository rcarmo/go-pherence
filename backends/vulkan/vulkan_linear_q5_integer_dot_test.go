package vulkan

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineQ5IntegerDotOwner(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockVK(t, &vkIntegerDotEnabled, true)
	memory := mockMemory(t)
	mockVK(t, &vkDestroyShaderModule, func(VkDevice, VkShaderModule, unsafe.Pointer) {})
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	mockVK(t, &vkCreateShaderModule, func(_ VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		if _, e := vkInspectSPIRVMode(unsafe.Slice((*uint32)(ptr), int(n/4)), true); e != nil {
			t.Fatal(e)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(_ VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(_ VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult { *out = 3; return VK_SUCCESS })
	mockVK(t, &vkCreateComputePipelines, func(_ VkDevice, _ uintptr, _ uint32, p, a unsafe.Pointer, out *VkPipeline) VkResult {
		*out = 4
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorPool, func(_ VkDevice, p, a unsafe.Pointer, out *VkDescriptorPool) VkResult { *out = 5; return VK_SUCCESS })
	mockVK(t, &vkAllocateDescriptorSets, func(_ VkDevice, p unsafe.Pointer, out *VkDescriptorSet) VkResult { *out = 6; return VK_SUCCESS })
	mockVK(t, &vkAllocateCommandBuffers, func(_ VkDevice, p unsafe.Pointer, out *VkCommandBuffer) VkResult { *out = 7; return VK_SUCCESS })
	mockVK(t, &vkCreateFence, func(_ VkDevice, p, a unsafe.Pointer, out *VkFence) VkResult { *out = 8; return VK_SUCCESS })
	op, e := NewVkLinearQ5IntegerDotHybridSetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { return make([]byte, 22), nil })
	if e != nil {
		t.Fatal(e)
	}
	if op.StorageBytes() != 24 || op.quant.numBuffers != 2 || op.weights.kernel.numBuffers != 4 {
		t.Fatal("owner")
	}
	memory.requirement = 4096
	memory.props.memoryHeaps[0].size = 1 << 20
	backing := new([4096]byte)
	mockVK(t, &vkMapMemory, func(_ VkDevice, _ VkDeviceMemory, _ uint64, _ uint64, _ uint32, out *unsafe.Pointer) VkResult {
		*out = unsafe.Pointer(&backing[0])
		return VK_SUCCESS
	})
	a, e := NewVkTensorArena(context.Background(), 4096)
	if e != nil {
		t.Fatal(e)
	}
	defer a.Close()
	alloc := func(shape ...int) *VkTensorF32 {
		x, e := a.AllocF32(context.Background(), shape...)
		if e != nil {
			t.Fatal(e)
		}
		return x
	}
	x, out, bias, q := alloc(1, 32), alloc(1, 1), alloc(1), alloc(9)
	stages, e := op.Stages(context.Background(), 0, out, x, bias, q)
	if e != nil || len(stages) != 2 || !reflect.DeepEqual(stages[0].PushWords, []uint32{1}) || !reflect.DeepEqual(stages[1].PushWords, []uint32{1, 32, 1}) {
		t.Fatal("stage", e)
	}
	f32, e := op.F32Stage(context.Background(), 0, out, x, bias)
	if e != nil || f32.Kernel != op.f32 || len(f32.bindings) != 4 || !reflect.DeepEqual(f32.PushWords, []uint32{1, 32, 1}) {
		t.Fatal("hybrid F32 stage", e)
	}
	if _, e = op.F32Stage(context.Background(), 1, out, x, bias); e == nil {
		t.Fatal("hybrid index")
	}
	if _, e = op.Stages(context.Background(), 0, out, x, bias, bias); e == nil {
		t.Fatal("scratch shape")
	}
	if _, e = op.Stages(context.Background(), 1, out, x, bias, q); e == nil {
		t.Fatal("index")
	}
	alias := *q
	alias.offset = x.offset
	if _, e = op.Stages(context.Background(), 0, out, x, bias, &alias); e == nil {
		t.Fatal("scratch overlap")
	}
	aliasOut := *out
	aliasOut.offset = x.offset
	if _, e = op.Stages(context.Background(), 0, &aliasOut, x, bias, q); e == nil {
		t.Fatal("output overlap")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e = op.Stages(ctx, 0, out, x, bias, q); !errors.Is(e, context.Canceled) {
		t.Fatal("cancel", e)
	}
	// Pending descriptors retain both shared weights and quantisation kernels.
	op.weights.kernel.device = k.device
	pending := &vkPendingSubmission{kernel: op.quant}
	old := vkPending
	vkPending = pending
	if e = op.Close(); !errors.Is(e, ErrVulkanInFlight) {
		t.Fatal("pending", e)
	}
	vkPending = &vkPendingSubmission{kernel: op.f32}
	if e = op.Close(); !errors.Is(e, ErrVulkanInFlight) {
		t.Fatal("hybrid pending", e)
	}
	vkPending = old
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = op.Stages(context.Background(), 0, out, x, bias, q); e == nil {
		t.Fatal("closed")
	}
	if _, e = op.F32Stage(context.Background(), 0, out, x, bias); e == nil {
		t.Fatal("closed hybrid")
	}
	for _, arm := range []string{"reader", "extent", "quant-create", "hybrid-create", "cancel-create"} {
		before := memory.frees
		created := 0
		mockVK(t, &vkCreateShaderModule, func(_ VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
			created++
			if (arm == "quant-create" && created == 2) || (arm == "hybrid-create" && created == 3) {
				return -3
			}
			*out = 1
			return VK_SUCCESS
		})
		ctx, cancel := context.WithCancel(context.Background())
		construct := NewVkLinearQ5IntegerDotSetStream
		if arm == "hybrid-create" {
			construct = NewVkLinearQ5IntegerDotHybridSetStream
		}
		original := vkCreateShaderModule
		if arm == "cancel-create" {
			mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
				r := original(d, p, a, out)
				if created == 2 {
					cancel()
				}
				return r
			})
		}
		owner, err := construct(ctx, []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) {
			if arm == "reader" {
				return nil, errors.New("reader failed")
			}
			if arm == "extent" {
				return make([]byte, 21), nil
			}
			return make([]byte, 22), nil
		})
		cancel()
		if owner != nil || err == nil || memory.frees != before+1 {
			t.Fatal("rollback", arm, owner, err, memory.frees, before)
		}
	}
	before := memory.frees
	mockVK(t, &vkIntegerDotEnabled, false)
	if o, e := NewVkLinearQ5IntegerDotSetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { t.Fatal("reader before feature"); return nil, nil }); o != nil || e == nil {
		t.Fatal("not enabled")
	}
	if memory.frees != before {
		t.Fatal("allocation before feature")
	}
}
