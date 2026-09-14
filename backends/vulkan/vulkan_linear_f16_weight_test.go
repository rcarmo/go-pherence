package vulkan

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
)

func unpackLinearF16Weights(words []uint32, count int) []float32 {
	out := make([]float32, count)
	for i := range out {
		out[i] = half.F16ToF32(uint16(words[i/2] >> (16 * uint(i&1))))
	}
	return out
}

func linearF16WeightTileModel(x []float32, packed []uint32, bias []float32, rows, inDim, outDim int) ([]float32, []int) {
	weight := unpackLinearF16Weights(packed, outDim*inDim)
	return linearTileModel(x, weight, bias, rows, inDim, outDim, 3)
}

func TestVulkanOfflineLinearF16WeightPackingAndModel(t *testing.T) {
	for _, count := range []int{1, 2, 3, 15, 16, 17, 255, 256, 257} {
		weight := make([]float32, count)
		for i := range weight {
			weight[i] = float32(math.Sin(float64(i)*.17)) * 12
		}
		packed, err := packLinearF16Weights(weight)
		if err != nil || len(packed) != (count+1)/2 {
			t.Fatal(count, len(packed), err)
		}
		got := unpackLinearF16Weights(packed, count)
		for i, v := range got {
			want := half.F16ToF32(half.F32ToF16(weight[i]))
			if math.Float32bits(v) != math.Float32bits(want) {
				t.Fatal("pack order/rounding", count, i, v, want)
			}
		}
		if count&1 != 0 && packed[len(packed)-1]>>16 != 0 {
			t.Fatal("odd padding not zero")
		}
	}
	for _, bad := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}, {65520}, {-65520}} {
		if packed, err := packLinearF16Weights(bad); err == nil || packed != nil {
			t.Fatal("invalid weight accepted", bad)
		}
	}
	for _, dims := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {17, 31, 19}, {3, 384, 9}, {2, 1280, 7}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, weight, bias := nativeData(m*k, 71, .5), nativeData(n*k, 72, .5), nativeData(n, 73, .1)
		packed, err := packLinearF16Weights(weight)
		if err != nil {
			t.Fatal(err)
		}
		got, writers := linearF16WeightTileModel(x, packed, bias, m, k, n)
		want := nativeLinearRef(x, unpackLinearF16Weights(packed, n*k), bias, m, k, n)
		for i, v := range got {
			if writers[i] != 1 || math.IsNaN(float64(v)) || math.Abs(float64(v)-want[i]) > 2e-5+2e-5*math.Abs(want[i]) {
				t.Fatal("packed source model", dims, i, v, want[i], writers[i])
			}
		}
	}
}

func TestVulkanOfflineLinearF16WeightBindingsAndLifetime(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	memory := mockMemory(t)
	kernel.numBuffers, kernel.pushSize = 4, 12
	weight, err := VkBufAlloc(8)
	if err != nil {
		t.Fatal(err)
	}
	words := unsafe.Slice((*uint32)(weight.mapped), 2)
	words[0], words[1] = 0x40003c00, 0x44004200
	op := &VkLinearF16WeightF32{kernel: kernel, weight: weight, inDim: 2, outDim: 2, weightValues: 4}
	a := mustArena(t, 64)
	x, bias, out := mustTensor(t, a, 2, 2), mustTensor(t, a, 2), mustTensor(t, a, 2, 2)
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
	if !reflect.DeepEqual(ranges, [][3]uint64{{arenaHandle, 0, 16}, {uint64(weight.buf), 0, 8}, {arenaHandle, 16, 8}, {arenaHandle, 32, 16}}) || !reflect.DeepEqual(push, []uint32{2, 2, 2}) || groups != ([3]uint32{1, 1, 1}) {
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

func TestVulkanOfflineLinearF16WeightAdmissionAndContract(t *testing.T) {
	contract, err := InspectVulkanShader(spirv_linear_f16_weight_f32)
	if err != nil || contract != (VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 2048, StorageBindings: 15, PushBytes: 12}) {
		t.Fatal(contract, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if op, err := NewVkLinearF16WeightF32(ctx, []float32{1}, 1, 1); op != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled construction", op, err)
	}
	if err := (*VkLinearF16WeightF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	var zero *VkLinearF16WeightF32
	if err := zero.Forward(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("nil forward")
	}
}
