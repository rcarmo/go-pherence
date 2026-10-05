package vulkan

import (
	"strings"
	"testing"
	"unsafe"
)

func TestVulkanDescriptorScratchReuse(t *testing.T) {
	old := vkUpdateDescriptorSets
	defer func() { vkUpdateDescriptorSets = old }()
	kernel := &VkComputeKernel{device: 7}
	buffer := &VkBuf{buf: 11}
	bindings := make([]vkBufferBinding, 16)
	for i := range bindings {
		bindings[i] = vkBufferBinding{buffer: buffer, offset: uint64(i * 4), size: 4}
	}
	set := VkDescriptorSet(23)
	vkUpdateDescriptorSets = func(device VkDevice, count uint32, ptr unsafe.Pointer, copies uint32, copyPtr unsafe.Pointer) {
		if device != kernel.device || count != uint32(len(bindings)) || copies != 0 || copyPtr != nil {
			t.Fatal("descriptor call changed")
		}
		writes := unsafe.Slice((*vkDescriptorWrite)(ptr), int(count))
		for i, write := range writes {
			info := (*vkDescriptorBufferInfo)(write.pBufferInfo)
			if write.dstSet != set || write.dstBinding != uint32(i) || write.descriptorCount != 1 || write.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || info != &kernel.descriptorBuffers[i] || info.buffer != buffer.buf || info.offset != bindings[i].offset || info.rng != bindings[i].size {
				t.Fatal("stale descriptor scratch", i)
			}
		}
	}
	kernel.updateBindingsLocked(set, bindings)
	for i := range bindings {
		bindings[i].offset += 64
	}
	set++
	if allocs := testing.AllocsPerRun(10, func() { kernel.updateBindingsLocked(set, bindings) }); allocs != 0 {
		t.Fatal("descriptor scratch allocates", allocs)
	}
}

func TestVulkanDispatchRejectsInvalidKernelAndInputs(t *testing.T) {
	validBuf := &VkBuf{buf: 1, mem: 1, size: 4}
	cases := []struct {
		name string
		k    *VkComputeKernel
		gx   uint32
		bufs []*VkBuf
		want string
	}{
		{"nil kernel", nil, 1, []*VkBuf{validBuf}, "uninitialized kernel"},
		{"zero pipeline", &VkComputeKernel{pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{validBuf}, "uninitialized kernel"},
		{"zero workgroups", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 0, []*VkBuf{validBuf}, "zero workgroups"},
		{"wrong buffer count", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 2}, 1, []*VkBuf{validBuf}, "buffer count"},
		{"nil buffer", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{nil}, "buffer 0"},
		{"zero buffer handle", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{{mem: 1, size: 4}}, "buffer 0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.k.Dispatch(tc.gx, 1, 1, tc.bufs, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}
