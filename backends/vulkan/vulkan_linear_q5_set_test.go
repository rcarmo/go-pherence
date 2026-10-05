package vulkan

import (
	"context"
	"errors"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

func TestVulkanOfflineQ5SetPreparation(t *testing.T) {
	raw := make([]byte, 22)
	for _, align := range []uint64{4, 16, 64, 256} {
		v, p, n, err := prepareQ5Set(context.Background(), []VkLinearQ5Matrix{{raw, 1, 32}, {raw, 1, 32}}, align)
		if err != nil || len(v) != 2 || len(p) != 2 || v[1].offset%align != 0 || n != v[1].offset+24 {
			t.Fatal(v, n, err)
		}
	}
	for _, m := range [][]VkLinearQ5Matrix{nil, make([]VkLinearQ5Matrix, 513), {{raw, 0, 32}}, {{raw, 1, 31}}, {{raw, 2, 32}}} {
		if _, _, _, err := prepareQ5Set(context.Background(), m, 4); err == nil {
			t.Fatal("admission")
		}
	}
	if _, _, _, err := prepareQ5Set(nil, []VkLinearQ5Matrix{{raw, 1, 32}}, 4); err == nil {
		t.Fatal("nil")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := prepareQ5Set(ctx, []VkLinearQ5Matrix{{raw, 1, 32}}, 4); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, _, _, err := prepareQ5Set(context.Background(), []VkLinearQ5Matrix{{raw, 1, 32}}, 6); err == nil {
		t.Fatal("alignment")
	}
}
func TestVulkanOfflineQ5SetPlanLifetime(t *testing.T) {
	lane, kernel, _, _ := newPlanMock(t)
	memory := mockMemory(t)
	kernel.numBuffers = 4
	kernel.pushSize = 12
	storage := mustMemoryBuffer(t)
	storage.size = 64
	set := &VkLinearQ5Set{kernel: kernel, storage: storage, bytes: 48, views: []vkQ5View{{1, 32, 0, 24}, {1, 32, 32, 24}}}
	// Metadata-only tensor views over one admitted mocked arena.
	memory.requirement = 256
	arena := mustArena(t, 64)
	arena.state.buffer.size = 256
	x := linearMetadataTensor(1, 1, 32)
	x.arena = arena.state
	x.offset = 0
	x.size = 128
	bias := linearMetadataTensor(1, 1)
	bias.arena = arena.state
	bias.offset = 128
	bias.size = 4
	out := linearMetadataTensor(1, 1, 1)
	out.arena = arena.state
	out.offset = 160
	out.size = 4
	// Native allocation is 256 bytes; metadata views intentionally cover it.
	stage, err := set.Stage(context.Background(), 1, out, x, bias)
	if err != nil {
		t.Fatal(err)
	}
	if len(stage.Tensors) != 0 || !reflect.DeepEqual(stage.PushWords, []uint32{1, 32, 1}) || stage.bindings[1].offset != 32 || stage.bindings[1].size != 24 {
		t.Fatal(stage)
	}
	plan, err := NewVkF32Plan(context.Background(), []VkF32Stage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if err = plan.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	lane.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	expectErrorIs(t, set.Close(), ErrVulkanInFlight)
	expectErrorIs(t, plan.Close(), ErrVulkanInFlight)
	lane.hook = nil
	if err = VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if err = plan.Close(); err != nil {
		t.Fatal(err)
	}
	if err = set.Close(); err != nil {
		t.Fatal(err)
	}
	if err = set.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = set.Stage(context.Background(), 0, out, x, bias); err == nil {
		t.Fatal("closed")
	}
	if err = arena.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineQ5SetConstruction(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{16, 16, 1}) || c.SharedBytes != 16384 || c.StorageBindings != 15 || c.PushBytes != 12 {
			t.Fatal("native shader", c, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(p, 20)) != 4 {
			t.Fatal("bindings")
		}
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult {
		r := *(*unsafe.Pointer)(unsafe.Add(p, 40))
		if *(*uint32)(unsafe.Add(r, 8)) != 12 {
			t.Fatal("push range")
		}
		*out = 3
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateComputePipelines, func(d VkDevice, c uintptr, n uint32, p, a unsafe.Pointer, out *VkPipeline) VkResult {
		*out = 4
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorPool, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorPool) VkResult { *out = 5; return VK_SUCCESS })
	mockVK(t, &vkAllocateDescriptorSets, func(d VkDevice, p unsafe.Pointer, out *VkDescriptorSet) VkResult { *out = 6; return VK_SUCCESS })
	mockVK(t, &vkAllocateCommandBuffers, func(d VkDevice, p unsafe.Pointer, out *VkCommandBuffer) VkResult { *out = 7; return VK_SUCCESS })
	mockVK(t, &vkCreateFence, func(d VkDevice, p, a unsafe.Pointer, out *VkFence) VkResult { *out = 8; return VK_SUCCESS })
	freedShader := 0
	mockVK(t, &vkDestroyShaderModule, func(VkDevice, VkShaderModule, unsafe.Pointer) { freedShader++ })
	memory := mockMemory(t)
	raw := make([]byte, 22)
	raw[0], raw[1] = 0, 0x24
	op, err := NewVkLinearQ5Set(context.Background(), []VkLinearQ5Matrix{{raw, 1, 32}})
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.storage == nil || op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 || freedShader != 1 {
		t.Fatal("construction")
	}
	if math.Float32bits(half.F16ToF32(uint16(*(*uint32)(op.storage.mapped)))) != math.Float32bits(half.F16ToF32(0x2400)) {
		t.Fatal("upload")
	}
	raw[1] = 0
	if *(*uint32)(op.storage.mapped) != 0x2400 {
		t.Fatal("retained input")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if op.StorageBytes() != 24 {
		t.Fatal("storage stats")
	}
	if (*VkLinearQ5Set)(nil).StorageBytes() != 0 {
		t.Fatal("nil stats")
	}
	if memory.frees != 1 {
		t.Fatal("cleanup count", memory.frees)
	}
	mockVK(t, &vkCreateBuffer, func(_ VkDevice, _, _ unsafe.Pointer, _ *VkBuffer) VkResult { return -2 })
	result, err := NewVkLinearQ5Set(context.Background(), []VkLinearQ5Matrix{{make([]byte, 22), 1, 32}})
	if result != nil || err == nil {
		t.Fatal("allocation rollback", result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if owner, e := NewVkLinearQ5Set(ctx, []VkLinearQ5Matrix{{raw, 1, 32}}); owner != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("cancel constructor", e)
	}
}
