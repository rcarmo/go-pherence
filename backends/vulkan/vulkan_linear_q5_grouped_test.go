package vulkan

import (
	"context"
	"encoding/binary"
	"errors"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"os"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

func TestVulkanOfflineQ5GroupedPacking(t *testing.T) {
	for _, count := range []int{1, 2, 63, 64, 65, 1025} {
		raw := make([]byte, count*22)
		for i := range raw {
			raw[i] = byte(i*37 + 19)
		}
		for b := 0; b < count; b++ {
			binary.LittleEndian.PutUint16(raw[b*22:], uint16(b%0x7c00))
		}
		before := append([]byte(nil), raw...)
		words, err := packLinearQ5Blocks(context.Background(), raw)
		if err != nil || len(words) != count*6 {
			t.Fatal(count, err)
		}
		if !reflect.DeepEqual(before, raw) {
			t.Fatal("mutated source")
		}
		for b := 0; b < count; b++ {
			if words[b*6]>>16 != 0 {
				t.Fatal("padding")
			}
			for j := 0; j < 22; j++ {
				index := j
				if index >= 2 {
					index += 2
				}
				got := byte(words[b*6+index/4] >> uint(index%4*8))
				if got != raw[b*22+j] {
					t.Fatal("packing", b, j)
				}
			}
		}
	}
	for _, raw := range [][]byte{nil, make([]byte, 21), make([]byte, 23)} {
		if w, err := packLinearQ5Blocks(context.Background(), raw); w != nil || err == nil {
			t.Fatal("block boundary")
		}
	}
	for _, h := range []uint16{0x7c00, 0xfc00, 0x7c01, 0xffff} {
		raw := make([]byte, 22)
		binary.LittleEndian.PutUint16(raw, h)
		if _, err := packLinearQ5Blocks(context.Background(), raw); err == nil {
			t.Fatal("nonfinite scale")
		}
	}
	if _, err := packLinearQ5Blocks(nil, make([]byte, 22)); err == nil {
		t.Fatal("nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := packLinearQ5Blocks(ctx, make([]byte, 22)); !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
}
func TestVulkanOfflineQ5GroupedIndependentValues(t *testing.T) {
	raw, err := os.ReadFile("testdata/q5-grouped-input.bin")
	if err != nil {
		t.Fatal(err)
	}
	oracle, err := os.ReadFile("testdata/q5-grouped-output.f32")
	if err != nil {
		t.Fatal(err)
	}
	words, err := packLinearQ5Blocks(context.Background(), raw)
	if err != nil || len(oracle) != len(raw)/22*32*4 {
		t.Fatal(err)
	}
	for b := 0; b < len(raw)/22; b++ {
		scale := half.F16ToF32(uint16(words[b*6]))
		high := words[b*6+1]
		for i := 0; i < 32; i++ {
			j := i % 16
			qbyte := byte(words[b*6+2+j/4] >> uint(j%4*8))
			q := uint32(qbyte & 15)
			if i >= 16 {
				q = uint32(qbyte >> 4)
			}
			q |= ((high >> uint(i)) & 1) << 4
			got := scale * float32(int(q)-16)
			want := binary.LittleEndian.Uint32(oracle[(b*32+i)*4:])
			if math.Float32bits(got) != want {
				t.Fatal("independent value", b, i, got, math.Float32frombits(want))
			}
		}
	}
}
func TestVulkanOfflineQ5GroupedAdmission(t *testing.T) {
	got, err := InspectVulkanShader(spirv_linear_q5_grouped_f32)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 16384, StorageBindings: 15, PushBytes: 12}
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if op, err := NewVkLinearQ5GroupedF32(ctx, make([]byte, 22), 1, 32); op != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("constructor cancel", err)
	}
	if op, err := NewVkLinearQ5GroupedF32(nil, nil, 1, 32); op != nil || err == nil {
		t.Fatal("nil context")
	}
	if err := (*VkLinearQ5GroupedF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	offlineVK(t)
	for _, shape := range [][2]int{{0, 32}, {1, 31}, {16385, 32}, {1, 16416}, {1, 64}} {
		if op, err := NewVkLinearQ5GroupedF32(context.Background(), make([]byte, 22), shape[0], shape[1]); op != nil || err == nil {
			t.Fatal("shape", shape)
		}
	}
	vkLimits.SharedMemoryBytes = 16383
	if op, err := NewVkLinearQ5GroupedF32(context.Background(), make([]byte, 22), 1, 32); op != nil || !errors.Is(err, ErrVulkanLimit) {
		t.Fatal("shared memory", err)
	}
}
func TestVulkanOfflineLinearQ5GroupedBindingsAndLifetime(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	memory := mockMemory(t)
	kernel.numBuffers, kernel.pushSize = 4, 12
	weight, err := VkBufAlloc(48)
	if err != nil {
		t.Fatal(err)
	}
	words := unsafe.Slice((*uint32)(weight.mapped), 2)
	words[0], words[1] = 0x40003c00, 0x44004200
	op := &VkLinearQ5GroupedF32{kernel: kernel, weight: weight, inDim: 32, outDim: 2, weightValues: 64}
	memory.requirement = 320
	backing := make([]byte, 320)
	mockVK(t, &vkMapMemory, func(_ VkDevice, _ VkDeviceMemory, _, n uint64, _ uint32, out *unsafe.Pointer) VkResult {
		if n > 320 {
			t.Fatal("mock bound")
		}
		*out = unsafe.Pointer(&backing[0])
		return VK_SUCCESS
	})
	a := mustArena(t, 320)
	x, bias, out := mustTensor(t, a, 2, 32), mustTensor(t, a, 2), mustTensor(t, a, 2, 2)
	var ranges [][3]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(_ VkDevice, n uint32, p unsafe.Pointer, _ uint32, _ unsafe.Pointer) {
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _, _, n uint32, p unsafe.Pointer) {
		if n != 12 {
			t.Fatal("push size")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 3)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, bias); err != nil {
		t.Fatal(err)
	}
	arenaHandle := uint64(a.state.buffer.buf)
	if !reflect.DeepEqual(ranges, [][3]uint64{{arenaHandle, 0, 256}, {uint64(weight.buf), 0, 48}, {arenaHandle, 256, 8}, {arenaHandle, 272, 16}}) || !reflect.DeepEqual(push, []uint32{2, 32, 2}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	if _, err := NewVkF32Plan(context.Background(), []VkF32Stage{{Kernel: kernel, Groups: groups, Tensors: []*VkTensorF32{x, bias, out}, PushWords: push}}); err == nil {
		t.Fatal("mixed packed operator admitted to F32 plan")
	}
	alias := *out
	alias.arena, alias.offset = x.arena, x.offset
	if err := op.Forward(context.Background(), &alias, x, bias); err == nil {
		t.Fatal("output alias accepted")
	}
	if err := op.Forward(nil, out, x, bias); err == nil {
		t.Fatal("nil forward context")
	}
	if err := (*VkLinearQ5GroupedF32)(nil).Forward(context.Background(), out, x, bias); err == nil {
		t.Fatal("nil operator")
	}
	if err := op.Forward(context.Background(), nil, x, bias); err == nil {
		t.Fatal("nil out")
	}
	if err := op.Forward(context.Background(), out, nil, bias); err == nil {
		t.Fatal("nil x")
	}
	if err := op.Forward(context.Background(), out, x, nil); err == nil {
		t.Fatal("nil bias")
	}
	bad := *x
	bad.rank = 1
	if err := op.Forward(context.Background(), out, &bad, bias); err == nil {
		t.Fatal("rank")
	}
	bad = *x
	bad.shape[1] = 31
	if err := op.Forward(context.Background(), out, &bad, bias); err == nil {
		t.Fatal("shape")
	}
	bad = *x
	bad.size -= 4
	if err := op.Forward(context.Background(), out, &bad, bias); err == nil {
		t.Fatal("storage")
	}
	ctx, cancel := context.WithCancel(context.Background())
	lane.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	err = op.Forward(ctx, out, x, bias)
	expectErrorIs(t, err, ErrVulkanInFlight)
	beforeCloseFrees := memory.frees
	expectErrorIs(t, op.Close(), ErrVulkanInFlight)
	if memory.frees != beforeCloseFrees {
		t.Fatal("pending owner freed")
	}
	lane.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	copyOwner := *op
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal("non-idempotent close", err)
	}
	expectErrorIs(t, copyOwner.Forward(context.Background(), out, x, bias), ErrVulkanClosed)
	if err := copyOwner.Close(); err != nil {
		t.Fatal("copied owner close", err)
	}
	if memory.frees != beforeCloseFrees+1 || op.kernel != nil || op.weight != nil {
		t.Fatal("owned weight cleanup", memory.frees, op)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineQ5GroupedConstruction(t *testing.T) {
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
	op, err := NewVkLinearQ5GroupedF32(context.Background(), raw, 1, 32)
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.weight == nil || op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 || freedShader != 1 {
		t.Fatal("construction")
	}
	if math.Float32bits(half.F16ToF32(uint16(*(*uint32)(op.weight.mapped)))) != math.Float32bits(half.F16ToF32(0x2400)) {
		t.Fatal("upload")
	}
	raw[1] = 0
	if *(*uint32)(op.weight.mapped) != 0x2400 {
		t.Fatal("retained input")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if memory.frees != 1 {
		t.Fatal("cleanup count", memory.frees)
	}
	mockVK(t, &vkCreateBuffer, func(_ VkDevice, _, _ unsafe.Pointer, _ *VkBuffer) VkResult { return -2 })
	result, err := NewVkLinearQ5GroupedF32(context.Background(), make([]byte, 22), 1, 32)
	if result != nil || err == nil {
		t.Fatal("allocation rollback", result, err)
	}
}
