package vulkan

import (
	"context"
	"encoding/binary"
	"errors"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"reflect"
	"testing"
	"unsafe"
)

type q5StreamCancelContext struct {
	context.Context
	calls, at int
}

func (c *q5StreamCancelContext) Err() error {
	c.calls++
	if c.calls >= c.at {
		return context.Canceled
	}
	return nil
}

func TestVulkanOfflineQ5StreamPacking(t *testing.T) {
	for _, count := range []int{1, 2, 64, 65, 1025} {
		raw := make([]byte, count*22)
		for i := range raw {
			raw[i] = byte(i*29 + 7)
		}
		for b := 0; b < count; b++ {
			binary.LittleEndian.PutUint16(raw[b*22:], uint16(b%0x7c00))
		}
		want, err := packLinearQ5Blocks(context.Background(), raw)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]uint32, count*6+2)
		dst[0], dst[len(dst)-1] = 0xdeadbeef, 0x87654321
		if err = packQ5Into(context.Background(), raw, dst[1:len(dst)-1]); err != nil || !reflect.DeepEqual(dst[1:len(dst)-1], want) {
			t.Fatal("pack", err)
		}
		if dst[0] != 0xdeadbeef || dst[len(dst)-1] != 0x87654321 {
			t.Fatal("sentinel")
		}
	}
	for _, raw := range [][]byte{nil, make([]byte, 21), make([]byte, 23)} {
		if err := packQ5Into(context.Background(), raw, make([]uint32, 6)); err == nil {
			t.Fatal("extent")
		}
	}
	raw := make([]byte, 22)
	if err := packQ5Into(nil, raw, make([]uint32, 6)); err == nil {
		t.Fatal("nil")
	}
	if err := packQ5Into(context.Background(), raw, make([]uint32, 5)); err == nil {
		t.Fatal("dst extent")
	}
	for _, scale := range []uint16{0x7c00, 0xfc00, 0x7e00} {
		binary.LittleEndian.PutUint16(raw, scale)
		if err := packQ5Into(context.Background(), raw, make([]uint32, 6)); err == nil {
			t.Fatal("nonfinite")
		}
	}
	final := &q5StreamCancelContext{Context: context.Background(), at: 3}
	if err := packQ5Into(final, make([]byte, 22), make([]uint32, 6)); !errors.Is(err, context.Canceled) {
		t.Fatal("final cancellation", err)
	}
	mid := &q5StreamCancelContext{Context: context.Background(), at: 3}
	if err := packQ5Into(mid, make([]byte, 1025*22), make([]uint32, 1025*6)); !errors.Is(err, context.Canceled) {
		t.Fatal("block cancellation", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := packQ5Into(ctx, raw, make([]uint32, 6)); !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
}
func TestVulkanOfflineQ5StreamGeometry(t *testing.T) {
	for _, align := range []uint64{4, 16, 64, 256} {
		views, n, err := describeQ5Stream(context.Background(), []VkLinearQ5Shape{{1, 32}, {3, 64}}, align)
		if err != nil || n != views[1].offset+144 || views[1].offset%align != 0 {
			t.Fatal(views, n, err)
		}
	}
	for _, shapes := range [][]VkLinearQ5Shape{nil, make([]VkLinearQ5Shape, 513), {{0, 32}}, {{1, 31}}, {{16385, 32}}, {{1, 16416}}} {
		if _, _, err := describeQ5Stream(context.Background(), shapes, 4); err == nil {
			t.Fatal("shape")
		}
	}
	huge := make([]VkLinearQ5Shape, 32)
	for i := range huge {
		huge[i] = VkLinearQ5Shape{16384, 16384}
	}
	if _, _, err := describeQ5Stream(context.Background(), huge, 4); err == nil {
		t.Fatal("aggregate")
	}
	if _, _, err := describeQ5Stream(nil, []VkLinearQ5Shape{{1, 32}}, 4); err == nil {
		t.Fatal("nil")
	}
	if _, _, err := describeQ5Stream(context.Background(), []VkLinearQ5Shape{{1, 32}}, 6); err == nil {
		t.Fatal("align")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := describeQ5Stream(ctx, []VkLinearQ5Shape{{1, 32}}, 4); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if owner, err := NewVkLinearQ5SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, nil); owner != nil || err == nil {
		t.Fatal("nil reader")
	}
}

func TestVulkanOfflineQ5StreamConstruction(t *testing.T) {
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
	op, err := NewVkLinearQ5SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { return raw, nil })
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
	for _, arm := range []string{"reader", "extent", "scale", "cancel"} {
		before := memory.frees
		ctx, stop := context.WithCancel(context.Background())
		calls := 0
		owner, e := NewVkLinearQ5SetStream(ctx, []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) {
			calls++
			switch arm {
			case "reader":
				return nil, errors.New("read failed")
			case "extent":
				return make([]byte, 21), nil
			case "scale":
				bad := make([]byte, 22)
				binary.LittleEndian.PutUint16(bad, 0x7c00)
				return bad, nil
			case "cancel":
				stop()
			}
			return make([]byte, 22), nil
		})
		stop()
		if owner != nil || e == nil || calls != 1 || memory.frees != before+1 {
			t.Fatal("reader rollback", arm, calls, e, memory.frees, before)
		}
	}
	beforePanic := memory.frees
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("reader panic not propagated")
			}
		}()
		_, _ = NewVkLinearQ5SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { panic("reader panic") })
	}()
	if memory.frees != beforePanic+1 {
		t.Fatal("reader panic leaked allocation")
	}
	var calls int
	if owner, e := NewVkLinearQ5SetStream(context.Background(), []VkLinearQ5Shape{{0, 32}}, func(context.Context, int) ([]byte, error) { calls++; return nil, nil }); owner != nil || e == nil || calls != 0 {
		t.Fatal("preflight before reader", e)
	}
	mockVK(t, &vkCreateBuffer, func(_ VkDevice, _, _ unsafe.Pointer, _ *VkBuffer) VkResult { return -2 })
	result, err := NewVkLinearQ5SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { return make([]byte, 22), nil })
	if result != nil || err == nil {
		t.Fatal("allocation rollback", result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if owner, e := NewVkLinearQ5SetStream(ctx, []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { return raw, nil }); owner != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("cancel constructor", e)
	}
}
