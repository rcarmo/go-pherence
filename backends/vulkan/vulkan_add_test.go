package vulkan

import (
	"context"
	"math"
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineAddBindings(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	k.numBuffers = 3
	k.pushSize = 4
	op := &VkAddF32{kernel: k}
	a := mustArena(t, 2048)
	x, y, out := mustTensor(t, a, 257), mustTensor(t, a, 3), mustTensor(t, a, 3)
	// Use small views for all alias combinations; large tensor checks tail grid.
	small := mustTensor(t, a, 3)
	xSmall := *small
	var ranges [][2]uint64
	var push uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, z unsafe.Pointer) {
		ranges = nil
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [2]uint64{*(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if n != 4 || off != 0 || s != 0x20 {
			t.Fatal("addpush")
		}
		push = *(*uint32)(p)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	for _, args := range [][3]*VkTensorF32{{out, small, y}, {small, small, y}, {y, small, y}, {small, small, small}, {out, small, small}, {&xSmall, small, y}, {x, x, x}} {
		dest, left, right := args[0], args[1], args[2]
		if err := op.Forward(context.Background(), dest, left, right); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ranges, [][2]uint64{{left.offset, left.size}, {right.offset, right.size}, {dest.offset, dest.size}}) || push != uint32(dest.Elements()) || groups != ([3]uint32{uint32((dest.Elements() + 255) / 256), 1, 1}) {
			t.Fatal(ranges, push, groups)
		}
	}
	s, err := op.Stage(context.Background(), out, small, y)
	if err != nil {
		t.Fatal(err)
	}
	s.PushWords[0] = 0
	s.Tensors[0] = nil
	again, err := op.Stage(context.Background(), out, small, y)
	if err != nil || again.PushWords[0] != 3 || again.Tensors[0] != small {
		t.Fatal("borrowedstage")
	}
	a.Close()
	op.Close()
}
func TestVulkanOfflineAddAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 4
	op := &VkAddF32{kernel: k}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("notdispatch")
	})
	for rank := 1; rank <= 8; rank++ {
		shape := make([]int, rank)
		for i := range shape {
			shape[i] = 1
		}
		a, b, out := linearMetadataTensor(1, shape...), linearMetadataTensor(2, shape...), linearMetadataTensor(3, shape...)
		if _, err := op.Stage(context.Background(), out, a, b); err != nil {
			t.Fatal(rank, err)
		}
	}
	a, b, out := linearMetadataTensor(1, 8), linearMetadataTensor(2, 8), linearMetadataTensor(3, 8)
	for slot := 0; slot < 3; slot++ {
		for _, mutate := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 0 }, func(v *VkTensorF32) { v.rank = 9 }, func(v *VkTensorF32) { v.shape[0] = 0 }, func(v *VkTensorF32) { v.shape[0] = 7 }, func(v *VkTensorF32) { v.size = 28 }} {
			args := []*VkTensorF32{out, a, b}
			bad := *args[slot]
			mutate(&bad)
			args[slot] = &bad
			if _, err := op.Stage(context.Background(), args[0], args[1], args[2]); err == nil {
				t.Fatal("badshape", slot)
			}
		}
	}
	for _, input := range []*VkTensorF32{a, b} {
		partial := *input
		partial.offset = 16
		input.arena.buffer.size = 48
		if _, err := op.Stage(context.Background(), &partial, a, b); err == nil {
			t.Fatal("partialoutput")
		}
	}
	// Exact first-input alias must not hide partial overlap with the other input.
	partial := *a
	partial.offset = 16
	if _, err := op.Stage(context.Background(), a, a, &partial); err == nil {
		t.Fatal("mixedalias")
	}
	big := linearMetadataTensor(4, 65536, 65536)
	if _, err := op.Stage(context.Background(), big, big, big); err == nil {
		t.Fatal("countoverflow")
	}
	vkLimits.StorageBufferRange = math.MaxUint32
	vkLimits.WorkgroupCount[0] = math.MaxUint32
	large := linearMetadataTensor(5, int(math.MaxUint32/4))
	s, err := op.Stage(context.Background(), large, large, large)
	if err != nil || s.Groups[0] != 4194304 {
		t.Fatal("maxrange", err)
	}
	larger := linearMetadataTensor(5, 1073741824)
	if _, err := op.Stage(context.Background(), larger, larger, larger); err == nil {
		t.Fatal("rangeoverflow")
	}
	vkLimits = offlineLimits()
	vkLimits.WorkgroupCount[0] = 1
	tail := linearMetadataTensor(6, 257)
	if _, err := op.Stage(context.Background(), tail, tail, tail); err == nil {
		t.Fatal("gridlimit")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = op.Stage(ctx, out, a, b)
	expectErrorIs(t, err, context.Canceled)
	if len(lane.events) != 0 {
		t.Fatal("nativeadmission")
	}
	for i := 0; i < 3; i++ {
		args := []*VkTensorF32{out, a, b}
		args[i] = nil
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2]); err == nil {
			t.Fatal("nil")
		}
	}
	if err := (*VkAddF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkAddF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkAddF32).Stage(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("zero")
	}
}
func TestVulkanOfflineAddContract(t *testing.T) {
	offlineVK(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewVkAddF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkAddF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}

func TestVulkanOfflineAddCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{256, 1, 1}) || c.SharedBytes != 0 || c.StorageBindings != 7 || c.PushBytes != 4 {
			t.Fatal("native shader", c, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(p, 20)) != 3 {
			t.Fatal("bindings")
		}
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult {
		r := *(*unsafe.Pointer)(unsafe.Add(p, 40))
		if *(*uint32)(unsafe.Add(r, 8)) != 4 {
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
	op, err := NewVkAddF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 3 || op.kernel.pushSize != 4 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
