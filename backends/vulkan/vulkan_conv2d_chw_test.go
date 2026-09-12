package vulkan

import (
	"context"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

func conv2DCHWReference(x, weight []float32, inChannels, inFrequency, inFrames, outChannels, kernel, stride, padding int) []float64 {
	outFrequency := (inFrequency + stride - 1) / stride
	outFrames := (inFrames + stride - 1) / stride
	out := make([]float64, outChannels*outFrequency*outFrames)
	for oc := 0; oc < outChannels; oc++ {
		for of := 0; of < outFrequency; of++ {
			for ot := 0; ot < outFrames; ot++ {
				var sum float64
				for ic := 0; ic < inChannels; ic++ {
					for kf := 0; kf < kernel; kf++ {
						for kt := 0; kt < kernel; kt++ {
							inf, intm := of*stride+kf-padding, ot*stride+kt-padding
							if inf >= 0 && inf < inFrequency && intm >= 0 && intm < inFrames {
								xi := (ic*inFrequency+inf)*inFrames + intm
								wi := ((oc*inChannels+ic)*kernel+kf)*kernel + kt
								sum += float64(x[xi]) * float64(weight[wi])
							}
						}
					}
				}
				out[(oc*outFrequency+of)*outFrames+ot] = sum
			}
		}
	}
	return out
}

func conv2DCHWTileModel(t *testing.T, x, weight []float32, inChannels, inFrequency, inFrames, outChannels, kernel, stride, padding int, seed int64) []float32 {
	t.Helper()
	outFrequency := (inFrequency + stride - 1) / stride
	outFrames := (inFrames + stride - 1) / stride
	outSpatial := outFrequency * outFrames
	reduction := inChannels * kernel * kernel
	groupsX, groupsY := (outChannels+15)/16, (outSpatial+15)/16
	out := make([]float32, outChannels*outSpatial+4)
	for i := outChannels * outSpatial; i < len(out); i++ {
		out[i] = 123
	}
	writers := make([]int, outChannels*outSpatial)
	rng := rand.New(rand.NewSource(seed))
	lanes := rng.Perm(256)
	for _, group := range rng.Perm(groupsX * groupsY) {
		groupX, groupY := group%groupsX, group/groupsX
		var sums, inputTile, weightTile [256]float32
		for base := 0; base < reduction; base += 16 {
			for _, lane := range lanes {
				cx, ry := lane%16, lane/16
				position, ri := groupY*16+ry, base+cx
				inputTile[lane], weightTile[lane] = 0, 0
				if position < outSpatial && ri < reduction {
					of, ot := position/outFrames, position%outFrames
					tapArea := kernel * kernel
					ic, tap := ri/tapArea, ri%tapArea
					inf, intm := of*stride+tap/kernel-padding, ot*stride+tap%kernel-padding
					if inf >= 0 && inf < inFrequency && intm >= 0 && intm < inFrames {
						inputTile[lane] = x[(ic*inFrequency+inf)*inFrames+intm]
					}
				}
				oc := groupX*16 + ry
				if oc < outChannels && ri < reduction {
					weightTile[lane] = weight[oc*reduction+ri]
				}
			}
			for _, lane := range lanes {
				cx, ry := lane%16, lane/16
				for j := 0; j < 16; j++ {
					sums[lane] = float32(sums[lane] + float32(inputTile[ry*16+j]*weightTile[cx*16+j]))
				}
			}
		}
		for _, lane := range lanes {
			cx, ry := lane%16, lane/16
			position, oc := groupY*16+ry, groupX*16+cx
			if position < outSpatial && oc < outChannels {
				index := oc*outSpatial + position
				out[index] = sums[lane]
				writers[index]++
			}
		}
	}
	for _, count := range writers {
		if count != 1 {
			t.Fatal("conv2d output owner", count)
		}
	}
	for _, value := range out[outChannels*outSpatial:] {
		if value != 123 {
			t.Fatal("conv2d tail guard")
		}
	}
	return out[:outChannels*outSpatial]
}

func TestVulkanOfflineConv2DCHWNumericsAndSchedule(t *testing.T) {
	dimensions := [][7]int{
		{1, 1, 1, 1, 1, 1, 0}, {1, 2, 3, 2, 3, 1, 1}, {2, 3, 5, 3, 3, 2, 1},
		{3, 17, 19, 5, 1, 1, 0}, {15, 6, 7, 17, 3, 2, 1}, {17, 5, 18, 15, 3, 1, 1},
		{31, 2, 33, 7, 1, 2, 0}, {64, 1, 9, 33, 3, 2, 1},
	}
	comparisons := 0
	maxError := 0.0
	for shapeIndex, shape := range dimensions {
		inChannels, frequency, frames, outChannels := shape[0], shape[1], shape[2], shape[3]
		kernel, stride, padding := shape[4], shape[5], shape[6]
		x := nativeData(inChannels*frequency*frames, int64(100+shapeIndex), .5)
		weight := nativeData(outChannels*inChannels*kernel*kernel, int64(200+shapeIndex), .5)
		reference := conv2DCHWReference(x, weight, inChannels, frequency, frames, outChannels, kernel, stride, padding)
		var prior []float32
		for seed := int64(0); seed < 4; seed++ {
			got := conv2DCHWTileModel(t, x, weight, inChannels, frequency, frames, outChannels, kernel, stride, padding, seed)
			if prior != nil && !reflect.DeepEqual(prior, got) {
				t.Fatal("conv2d schedule mismatch", shape, seed)
			}
			prior = got
			for i, value := range got {
				err := math.Abs(float64(value) - reference[i])
				maxError = math.Max(maxError, err)
				if math.IsNaN(float64(value)) || err > 2e-5+2e-5*math.Abs(reference[i]) {
					t.Fatal("conv2d numeric envelope", shape, i, value, reference[i], err)
				}
				comparisons++
			}
		}
	}
	// Non-symmetric 3x3 taps establish channel/tap orientation and padding.
	x := []float32{1, 2, 3, 4}
	weight := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	got := conv2DCHWTileModel(t, x, weight, 1, 2, 2, 1, 3, 1, 1, 0)
	if !reflect.DeepEqual(got, []float32{77, 67, 47, 37}) {
		t.Fatal("conv2d orientation", got)
	}
	t.Logf("conv2d shapes=%d schedules=4 values=%d max_abs=%g", len(dimensions), comparisons, maxError)
}

func conv2DMockTensors(t *testing.T) (*VkTensorArena, *VkTensorF32, *VkTensorF32, *VkTensorF32) {
	arena := mustArena(t, 2048)
	return arena, mustTensor(t, arena, 3, 2, 3), mustTensor(t, arena, 2, 3, 5), mustTensor(t, arena, 3, 2, 3, 3)
}

func TestVulkanOfflineConv2DCHWBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	kernel.numBuffers, kernel.pushSize = 3, 36
	op := &VkConv2DCHWF32{kernel: kernel}
	arena, out, x, weight := conv2DMockTensors(t)
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
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, stage, offset, size uint32, p unsafe.Pointer) {
		if stage != 0x20 || offset != 0 || size != 36 {
			t.Fatal("conv2d push ABI")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 9)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, weight, 3, 2, 1); err != nil {
		t.Fatal(err)
	}
	wantRanges := [][2]uint64{{x.offset, x.size}, {weight.offset, weight.size}, {out.offset, out.size}}
	wantPush := []uint32{2, 3, 5, 3, 2, 3, 3, 2, 1}
	if !reflect.DeepEqual(ranges, wantRanges) || !reflect.DeepEqual(push, wantPush) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	stage, err := op.Stage(context.Background(), out, x, weight, 3, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	stage.PushWords[0], stage.Tensors[0] = 99, nil
	again, err := op.Stage(context.Background(), out, x, weight, 3, 2, 1)
	if err != nil || again.PushWords[0] != 2 || again.Tensors[0] != x {
		t.Fatal("borrowed conv2d stage")
	}
	arena.Close()
	op.Close()
}

func TestVulkanOfflineConv2DCHWAdmission(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers, kernel.pushSize = 3, 36
	op := &VkConv2DCHWF32{kernel: kernel}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("conv2d admission dispatched")
	})
	for _, geometry := range [][3]int{{1, 1, 0}, {1, 2, 0}, {3, 1, 1}, {3, 2, 1}} {
		kernelSize, stride, padding := geometry[0], geometry[1], geometry[2]
		x := linearMetadataTensor(1, 256, 80, 204)
		weight := linearMetadataTensor(2, 256, 256, kernelSize, kernelSize)
		out := linearMetadataTensor(3, 256, (80+stride-1)/stride, (204+stride-1)/stride)
		stage, err := op.Stage(context.Background(), out, x, weight, kernelSize, stride, padding)
		if err != nil || stage.Groups[0] != 16 || stage.Groups[1] != uint32((((80+stride-1)/stride)*((204+stride-1)/stride)+15)/16) {
			t.Fatal("valid Community-1 geometry", geometry, err)
		}
	}
	x := linearMetadataTensor(1, 2, 3, 5)
	weight := linearMetadataTensor(2, 3, 2, 3, 3)
	out := linearMetadataTensor(3, 3, 2, 3)
	for slot := 0; slot < 3; slot++ {
		args := []*VkTensorF32{out, x, weight}
		args[slot] = nil
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2], 3, 2, 1); err == nil {
			t.Fatal("nil conv2d tensor", slot)
		}
	}
	for _, geometry := range [][3]int{{0, 1, 0}, {2, 1, 1}, {3, 0, 1}, {3, 3, 1}, {3, 1, 0}, {1, 1, 1}} {
		if _, err := op.Stage(context.Background(), out, x, weight, geometry[0], geometry[1], geometry[2]); err == nil {
			t.Fatal("bad conv2d geometry", geometry)
		}
	}
	for slot := 0; slot < 3; slot++ {
		args := []*VkTensorF32{out, x, weight}
		bad := *args[slot]
		bad.rank--
		args[slot] = &bad
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2], 3, 2, 1); err == nil {
			t.Fatal("bad conv2d rank", slot)
		}
	}
	for _, badWeight := range []*VkTensorF32{linearMetadataTensor(2, 3, 3, 3, 3), linearMetadataTensor(2, 3, 2, 1, 1), linearMetadataTensor(2, 3, 2, 3, 2)} {
		if _, err := op.Stage(context.Background(), out, x, badWeight, 3, 2, 1); err == nil {
			t.Fatal("bad conv2d weight")
		}
	}
	for _, badOut := range []*VkTensorF32{linearMetadataTensor(3, 4, 2, 3), linearMetadataTensor(3, 3, 3, 3), linearMetadataTensor(3, 3, 2, 4)} {
		if _, err := op.Stage(context.Background(), badOut, x, weight, 3, 2, 1); err == nil {
			t.Fatal("bad conv2d output")
		}
	}
	for _, source := range []*VkTensorF32{x, weight} {
		alias := *out
		alias.arena = source.arena
		alias.arena.buffer.size = 4096
		if _, err := op.Stage(context.Background(), &alias, x, weight, 3, 2, 1); err == nil {
			t.Fatal("conv2d output alias")
		}
	}
	partial := *out
	partial.arena = x.arena
	partial.offset = 4
	if _, err := op.Stage(context.Background(), &partial, x, weight, 3, 2, 1); err == nil {
		t.Fatal("conv2d partial alias")
	}
	badStorage := *out
	badStorage.size -= 4
	if _, err := op.Stage(context.Background(), &badStorage, x, weight, 3, 2, 1); err == nil {
		t.Fatal("conv2d storage mismatch")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x, weight, 3, 2, 1)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[1] = 1
	largeX := linearMetadataTensor(4, 2, 80, 4096)
	largeWeight := linearMetadataTensor(5, 3, 2, 3, 3)
	largeOut := linearMetadataTensor(6, 3, 80, 4096)
	if _, err := op.Stage(context.Background(), largeOut, largeX, largeWeight, 3, 1, 1); err == nil {
		t.Fatal("conv2d grid bound")
	}
	if len(lane.events) != 0 {
		t.Fatal("conv2d admission invoked native", lane.events)
	}
	if err := (*VkConv2DCHWF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkConv2DCHWF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkConv2DCHWF32).Stage(context.Background(), nil, nil, nil, 1, 1, 0); err == nil {
		t.Fatal("zero conv2d operator")
	}
}

func TestVulkanOfflineConv2DCHWPlanAndContract(t *testing.T) {
	mock, kernel, oldArena, _ := newPlanMock(t)
	oldArena.Close()
	mockAttentionMemory(t)
	arena := mustArena(t, 2048)
	kernel.numBuffers, kernel.pushSize = 3, 36
	op := &VkConv2DCHWF32{kernel: kernel}
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _ uint32, offset, size uint32, _ unsafe.Pointer) {
		if offset != 0 || size != 36 {
			t.Error("conv2d plan push ABI")
		}
	})
	x := mustTensor(t, arena, 1, 2, 2)
	weight := mustTensor(t, arena, 1, 1, 3, 3)
	out := mustTensor(t, arena, 1, 2, 2)
	stage, err := op.Stage(context.Background(), out, x, weight, 3, 1, 1)
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

	offlineVK(t)
	contract, err := InspectVulkanShader(spirv_conv2d_chw_f32)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 2048, StorageBindings: 7, PushBytes: 36}
	if err != nil || contract != want {
		t.Fatal(contract, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	_, err = NewVkConv2DCHWF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.SharedMemoryBytes = 2047
	_, err = NewVkConv2DCHWF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}

func TestVulkanOfflineConv2DCHWCreation(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(_ VkDevice, info, _ unsafe.Pointer, out *VkShaderModule) VkResult {
		size := *(*uint64)(unsafe.Add(info, 24))
		code := *(*unsafe.Pointer)(unsafe.Add(info, 32))
		contract, err := vkInspectSPIRV(unsafe.Slice((*uint32)(code), int(size)/4))
		want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 2048, StorageBindings: 7, PushBytes: 36}
		if err != nil || contract != want {
			t.Fatal("conv2d native shader", contract, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(_ VkDevice, info, _ unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(info, 20)) != 3 {
			t.Fatal("conv2d descriptor bindings")
		}
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(_ VkDevice, info, _ unsafe.Pointer, out *VkPipelineLayout) VkResult {
		rangeInfo := *(*unsafe.Pointer)(unsafe.Add(info, 40))
		if *(*uint32)(unsafe.Add(rangeInfo, 8)) != 36 {
			t.Fatal("conv2d push range")
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
	op, err := NewVkConv2DCHWF32(context.Background())
	if err != nil || op == nil || op.kernel.numBuffers != 3 || op.kernel.pushSize != 36 {
		t.Fatal("conv2d construction", err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
