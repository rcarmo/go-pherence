package vulkan

import (
	"context"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func channelAffineModel(x, scale, shift []float32, channels int, relu bool) []float32 {
	out := make([]float32, len(x))
	spatial := len(x) / channels
	for i, value := range x {
		channel := i / spatial
		value = simd.FMA32Scalar(value, scale[channel], shift[channel])
		if relu {
			value = max(value, float32(0))
		}
		out[i] = value
	}
	return out
}

func TestVulkanOfflineChannelAffineNumericsAndSchedule(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	comparisons := 0
	for _, shape := range [][3]int{{1, 1, 1}, {2, 3, 5}, {3, 1, 257}, {32, 10, 17}, {256, 1, 3}} {
		channels, frequency, frames := shape[0], shape[1], shape[2]
		x := make([]float32, channels*frequency*frames)
		scale, shift := make([]float32, channels), make([]float32, channels)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 4)
		}
		for i := range scale {
			scale[i] = float32(rng.NormFloat64())
			shift[i] = float32(rng.NormFloat64())
		}
		for _, relu := range []bool{false, true} {
			want := channelAffineModel(x, scale, shift, channels, relu)
			for seed := int64(0); seed < 4; seed++ {
				got := make([]float32, len(x))
				writers := make([]int, len(x))
				for _, i := range rand.New(rand.NewSource(seed)).Perm(((len(x) + 255) / 256) * 256) {
					if i >= len(x) {
						continue
					}
					channel := i / (frequency * frames)
					value := simd.FMA32Scalar(x[i], scale[channel], shift[channel])
					if relu {
						value = max(value, float32(0))
					}
					got[i], writers[i] = value, writers[i]+1
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatal("schedule mismatch", shape, relu, seed)
				}
				for _, writes := range writers {
					if writes != 1 {
						t.Fatal("output ownership")
					}
				}
				comparisons += len(got)
			}
		}
	}
	// The shader's native F32 fma is not Go's widened float64 FMA followed by
	// conversion. Keep a known midpoint adversary so the schedule model cannot
	// silently regress to the double-rounded expression used by the current CPU
	// Community-1 BatchNorm path.
	a, b, c := math.Float32frombits(0x3f800001), float32(1.5), math.Float32frombits(0x80000001)
	exact := simd.FMA32Scalar(a, b, c)
	widened := float32(math.FMA(float64(a), float64(b), float64(c)))
	if math.Float32bits(exact) == math.Float32bits(widened) {
		t.Fatal("F32 FMA midpoint adversary collapsed")
	}
	got := channelAffineModel([]float32{a}, []float32{b}, []float32{c}, 1, false)
	if len(got) != 1 || math.Float32bits(got[0]) != math.Float32bits(exact) {
		t.Fatal("schedule model is not single-round F32 FMA")
	}
	// ReLU follows Go max(value,+0): both signed zero inputs become +0.
	zeros := channelAffineModel([]float32{0, float32(math.Copysign(0, -1))}, []float32{1}, []float32{0}, 1, true)
	if math.Float32bits(zeros[0]) != 0 || math.Float32bits(zeros[1]) != 0 {
		t.Fatal("signed zero ReLU", zeros)
	}
	t.Logf("channel affine schedule comparisons=%d", comparisons)
}

func TestVulkanOfflineChannelAffineBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	kernel.numBuffers, kernel.pushSize = 4, 12
	op := &VkChannelAffineReLUF32{kernel: kernel}
	arena := mustArena(t, 2048)
	x := mustTensor(t, arena, 2, 3, 5)
	scale, shift := mustTensor(t, arena, 2), mustTensor(t, arena, 2)
	out := mustTensor(t, arena, 2, 3, 5)
	var ranges [][2]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(_ VkDevice, n uint32, p unsafe.Pointer, _ uint32, _ unsafe.Pointer) {
		ranges = nil
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [2]uint64{*(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, stage, off, n uint32, p unsafe.Pointer) {
		if stage != 0x20 || off != 0 || n != 12 {
			t.Fatal("push ABI")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 3)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, scale, shift, true); err != nil {
		t.Fatal(err)
	}
	wantRanges := [][2]uint64{{x.offset, x.size}, {scale.offset, scale.size}, {shift.offset, shift.size}, {out.offset, out.size}}
	if !reflect.DeepEqual(ranges, wantRanges) || !reflect.DeepEqual(push, []uint32{2, 15, 1}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	if err := op.Forward(context.Background(), x, x, scale, shift, false); err != nil {
		t.Fatal("exact alias", err)
	}
	if !reflect.DeepEqual(push, []uint32{2, 15, 0}) || ranges[0] != ranges[3] {
		t.Fatal("alias/flag")
	}
	stage, err := op.Stage(context.Background(), out, x, scale, shift, true)
	if err != nil {
		t.Fatal(err)
	}
	stage.PushWords[0], stage.Tensors[0] = 99, nil
	again, err := op.Stage(context.Background(), out, x, scale, shift, true)
	if err != nil || !reflect.DeepEqual(again.PushWords, []uint32{2, 15, 1}) || again.Tensors[0] != x {
		t.Fatal("borrowed stage")
	}
	arena.Close()
	op.Close()
}

func TestVulkanOfflineChannelAffineAdmission(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers, kernel.pushSize = 4, 12
	op := &VkChannelAffineReLUF32{kernel: kernel}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("not dispatch")
	})
	for rank := 2; rank <= 8; rank++ {
		shape := make([]int, rank)
		shape[0] = 3
		for i := 1; i < rank; i++ {
			shape[i] = 1
		}
		x, out := linearMetadataTensor(1, shape...), linearMetadataTensor(2, shape...)
		scale, shift := linearMetadataTensor(3, 3), linearMetadataTensor(4, 3)
		stage, err := op.Stage(context.Background(), out, x, scale, shift, rank%2 == 0)
		if err != nil || stage.PushWords[0] != 3 || stage.PushWords[1] != 1 {
			t.Fatal("valid rank", rank, stage, err)
		}
	}
	x, out := linearMetadataTensor(1, 2, 3), linearMetadataTensor(2, 2, 3)
	scale, shift := linearMetadataTensor(3, 2), linearMetadataTensor(4, 2)
	for slot := 0; slot < 4; slot++ {
		args := []*VkTensorF32{out, x, scale, shift}
		args[slot] = nil
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2], args[3], false); err == nil {
			t.Fatal("nil", slot)
		}
	}
	for _, mutate := range []func(*VkTensorF32){
		func(v *VkTensorF32) { v.rank = 1 }, func(v *VkTensorF32) { v.rank = 9 }, func(v *VkTensorF32) { v.shape[0] = 3 }, func(v *VkTensorF32) { v.shape[1] = 4 }, func(v *VkTensorF32) { v.size = 20 },
	} {
		bad := *out
		mutate(&bad)
		if _, err := op.Stage(context.Background(), &bad, x, scale, shift, false); err == nil {
			t.Fatal("bad output")
		}
	}
	for _, coefficient := range []*VkTensorF32{scale, shift} {
		for _, mutate := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 2 }, func(v *VkTensorF32) { v.shape[0] = 3 }, func(v *VkTensorF32) { v.size = 4 }} {
			bad := *coefficient
			mutate(&bad)
			if _, err := op.Stage(context.Background(), out, x, &bad, shift, false); err == nil && coefficient == scale {
				t.Fatal("bad scale")
			}
			if _, err := op.Stage(context.Background(), out, x, scale, &bad, false); err == nil && coefficient == shift {
				t.Fatal("bad shift")
			}
		}
	}
	// Partial X/Out overlap and all coefficient/output overlap are forbidden.
	partial := *x
	partial.offset = 4
	x.arena.buffer.size = 28
	if _, err := op.Stage(context.Background(), &partial, x, scale, shift, false); err == nil {
		t.Fatal("partial output")
	}
	coefficientOut := *scale
	coefficientOut.rank, coefficientOut.shape[0], coefficientOut.shape[1], coefficientOut.size = 2, 2, 1, 8
	if _, err := op.Stage(context.Background(), &coefficientOut, &coefficientOut, scale, shift, false); err == nil {
		t.Fatal("output coefficient overlap")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x, scale, shift, false)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[0] = 1
	largeX, largeOut := linearMetadataTensor(8, 2, 129), linearMetadataTensor(9, 2, 129)
	if _, err := op.Stage(context.Background(), largeOut, largeX, scale, shift, false); err == nil {
		t.Fatal("grid limit")
	}
	if len(lane.events) != 0 {
		t.Fatal("admission invoked native")
	}
	if err := (*VkChannelAffineReLUF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkChannelAffineReLUF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkChannelAffineReLUF32).Stage(context.Background(), nil, nil, nil, nil, false); err == nil {
		t.Fatal("zero operator")
	}
}

func TestVulkanOfflineChannelAffinePlan(t *testing.T) {
	mock, kernel, arena, _ := newPlanMock(t)
	kernel.numBuffers, kernel.pushSize = 4, 12
	op := &VkChannelAffineReLUF32{kernel: kernel}
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _ uint32, off, n uint32, _ unsafe.Pointer) {
		if off != 0 || n != 12 {
			t.Error("channel affine plan push ABI")
		}
	})
	x := mustTensor(t, arena, 1, 1)
	scale, shift := mustTensor(t, arena, 1), mustTensor(t, arena, 1)
	stage, err := op.Stage(context.Background(), x, x, scale, shift, true)
	if err != nil {
		t.Fatal(err)
	}
	plan := mustPlan(t, stage)
	ctx, cancel := context.WithCancel(context.Background())
	mock.hook = func(event string) {
		if event == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	expectErrorIs(t, op.Close(), ErrVulkanInFlight)
	expectErrorIs(t, arena.Close(), ErrVulkanInFlight)
	mock.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	op.Close()
	arena.Close()
}

func TestVulkanOfflineChannelAffineContractAndCreation(t *testing.T) {
	offlineVK(t)
	contract, err := InspectVulkanShader(spirv_channel_affine_relu_f32)
	if err != nil || contract != (VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, StorageBindings: 15, PushBytes: 12}) {
		t.Fatal(contract, err)
	}
	// Fma is the only admitted ternary extended instruction; nearby IDs fail.
	words := spirvTestWords(spirv_channel_affine_relu_f32)
	fma, greater, notEqual := 0, 0, 0
	for i := 5; i < len(words); {
		n, opcode := int(words[i]>>16), uint16(words[i])
		if opcode == 12 && n == 8 && words[i+4] == 50 {
			fma++
		}
		if opcode == 186 {
			greater++
		}
		if opcode == 171 {
			notEqual++
		}
		i += n
	}
	if fma != 1 || greater != 1 || notEqual != 1 {
		t.Fatal("shader arithmetic/control envelope", fma, greater, notEqual)
	}
	for _, id := range []uint32{49, 51, 999} {
		bad := editSPIRV(spirv_channel_affine_relu_f32, 12, func(words []uint32) {
			if len(words) == 8 {
				words[4] = id
			}
		})
		_, err := InspectVulkanShader(bad)
		expectErrorIs(t, err, ErrVulkanShaderContract)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkChannelAffineReLUF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkChannelAffineReLUF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)

	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(_ VkDevice, p, _ unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		got, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || got.StorageBindings != 15 || got.PushBytes != 12 {
			t.Fatal(got, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(_ VkDevice, p, _ unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(p, 20)) != 4 {
			t.Fatal("bindings")
		}
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(_ VkDevice, p, _ unsafe.Pointer, out *VkPipelineLayout) VkResult {
		rangeInfo := *(*unsafe.Pointer)(unsafe.Add(p, 40))
		if *(*uint32)(unsafe.Add(rangeInfo, 8)) != 12 {
			t.Fatal("push")
		}
		*out = 3
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateComputePipelines, func(_ VkDevice, _ uintptr, _ uint32, _ unsafe.Pointer, _ unsafe.Pointer, out *VkPipeline) VkResult {
		*out = 4
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorPool, func(_ VkDevice, _ unsafe.Pointer, _ unsafe.Pointer, out *VkDescriptorPool) VkResult {
		*out = 5
		return VK_SUCCESS
	})
	mockVK(t, &vkAllocateDescriptorSets, func(_ VkDevice, _ unsafe.Pointer, out *VkDescriptorSet) VkResult { *out = 6; return VK_SUCCESS })
	mockVK(t, &vkAllocateCommandBuffers, func(_ VkDevice, _ unsafe.Pointer, out *VkCommandBuffer) VkResult { *out = 7; return VK_SUCCESS })
	mockVK(t, &vkCreateFence, func(_ VkDevice, _ unsafe.Pointer, _ unsafe.Pointer, out *VkFence) VkResult {
		*out = 8
		return VK_SUCCESS
	})
	mockVK(t, &vkDestroyShaderModule, func(VkDevice, VkShaderModule, unsafe.Pointer) {})
	op, err := NewVkChannelAffineReLUF32(context.Background())
	if err != nil || op == nil || op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 {
		t.Fatal("construction", err)
	}
	op.Close()
}
