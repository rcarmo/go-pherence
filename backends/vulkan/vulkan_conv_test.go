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

// Direct float64 conv, separately structured from implicit-patch tile model.
func conv3Reference(x, w, b []float32, length, in, out, stride int, layout VkConvInputLayout) []float64 {
	rows := (length + stride - 1) / stride
	y := make([]float64, rows*out)
	for oc := 0; oc < out; oc++ {
		for r := 0; r < rows; r++ {
			s := float64(b[oc])
			for ic := 0; ic < in; ic++ {
				for tap := 0; tap < 3; tap++ {
					pos := r*stride + tap - 1
					if pos >= 0 && pos < length {
						idx := ic*length + pos
						if layout == VkConvTimeMajor {
							idx = pos*in + ic
						}
						s += float64(x[idx]) * float64(w[(oc*in+ic)*3+tap])
					}
				}
			}
			y[r*out+oc] = s
		}
	}
	return y
}
func conv3TileModel(t *testing.T, x, w, b []float32, length, in, out, stride int, layout VkConvInputLayout, seed int64) []float32 {
	t.Helper()
	rows := (length + stride - 1) / stride
	gx, gy := (out+15)/16, (rows+15)/16
	y := make([]float32, rows*out+4)
	for i := rows * out; i < len(y); i++ {
		y[i] = 123
	}
	writers := make([]int, rows*out)
	rng := rand.New(rand.NewSource(seed))
	lanes := rng.Perm(256)
	for _, group := range rng.Perm(gx * gy) {
		bx, by := group%gx, group/gx
		var sum, xt, wt [256]float32
		for base := 0; base < in*3; base += 16 {
			for _, lane := range lanes {
				cx, ry := lane%16, lane/16
				row, ix := by*16+ry, base+cx
				xt[lane] = 0
				wt[lane] = 0
				if row < rows && ix < in*3 {
					channel, tap := ix/3, ix%3
					padded := row*stride + tap
					if padded > 0 && padded-1 < length {
						pos := padded - 1
						idx := channel*length + pos
						if layout == VkConvTimeMajor {
							idx = pos*in + channel
						}
						xt[lane] = x[idx]
					}
				}
				oc := bx*16 + ry
				if oc < out && ix < in*3 {
					wt[lane] = w[oc*in*3+ix]
				}
			}
			for _, lane := range lanes {
				cx, ry := lane%16, lane/16
				for j := 0; j < 16; j++ {
					sum[lane] = float32(sum[lane] + float32(xt[ry*16+j]*wt[cx*16+j]))
				}
			}
		}
		for _, lane := range lanes {
			cx, ry := lane%16, lane/16
			row, col := by*16+ry, bx*16+cx
			if row < rows && col < out {
				idx := row*out + col
				y[idx] = sum[lane] + b[col]
				writers[idx]++
			}
		}
	}
	for _, n := range writers {
		if n != 1 {
			t.Fatal("conv output owner", n)
		}
	}
	for _, n := range y[rows*out:] {
		if n != 123 {
			t.Fatal("conv tail guard")
		}
	}
	return y[:rows*out]
}
func TestVulkanOfflineConvNumerics(t *testing.T) {
	dims := [][4]int{{1, 1, 1, 1}, {1, 2, 3, 2}, {2, 3, 2, 2}, {15, 5, 17, 1}, {16, 6, 15, 2}, {17, 17, 19, 1}, {33, 31, 7, 2}, {65, 80, 17, 2}, {9, 128, 33, 1}, {3, 2048, 2, 2}, {4096, 1, 1, 1}}
	count, maxErr := 0, 0.
	for _, s := range dims {
		length, in, out, stride := s[0], s[1], s[2], s[3]
		cf, w, b := nativeData(length*in, 5, .5), nativeData(out*in*3, 6, .5), nativeData(out, 7, .1)
		tm := make([]float32, len(cf))
		for ic := 0; ic < in; ic++ {
			for pos := 0; pos < length; pos++ {
				tm[pos*in+ic] = cf[ic*length+pos]
			}
		}
		var prior []float32
		for _, layout := range []VkConvInputLayout{VkConvChannelsFirst, VkConvTimeMajor} {
			x := cf
			if layout == VkConvTimeMajor {
				x = tm
			}
			ref := conv3Reference(x, w, b, length, in, out, stride, layout)
			for seed := int64(0); seed < 4; seed++ {
				got := conv3TileModel(t, x, w, b, length, in, out, stride, layout, seed)
				if prior != nil && !reflect.DeepEqual(prior, got) {
					t.Fatal("layout/schedule mismatch", s)
				}
				prior = got
				for i, v := range got {
					err := math.Abs(float64(v) - ref[i])
					maxErr = math.Max(maxErr, err)
					count++
					if math.IsNaN(float64(v)) || err > 2e-5+2e-5*math.Abs(ref[i]) {
						t.Fatal(s, layout, i, v, ref[i], err)
					}
				}
			}
		}
	}
	t.Logf("conv shapes=%d layouts=2 schedules=4 values=%d max_abs=%g", len(dims), count, maxErr)
	// Analytic padding/stride/orientation: non-symmetric taps make reversal fail.
	ref := conv3TileModel(t, []float32{1, 2, 3}, []float32{10, 20, 30}, []float32{1}, 3, 1, 1, 1, VkConvChannelsFirst, 0)
	if !reflect.DeepEqual(ref, []float32{81, 141, 81}) {
		t.Fatal(ref)
	}
}
func convMockTensors(t *testing.T) (*VkTensorArena, *VkTensorF32, *VkTensorF32, *VkTensorF32, *VkTensorF32) {
	a := mustArena(t, 2048)
	return a, mustTensor(t, a, 3, 3), mustTensor(t, a, 2, 5), mustTensor(t, a, 3, 2, 3), mustTensor(t, a, 3)
}
func TestVulkanOfflineConvBindings(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	k.numBuffers = 4
	k.pushSize = 24
	op := &VkConv1D3F32{kernel: k}
	a, out, x, w, b := convMockTensors(t)
	var ranges [][3]uint64
	var push []uint32
	var grid [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, z unsafe.Pointer) {
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if n != 24 || off != 0 || s != 0x20 {
			t.Fatal("conv push")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 6)...)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { grid = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, w, b, 2, VkConvChannelsFirst); err != nil {
		t.Fatal(err)
	}
	expected := [][3]uint64{}
	for _, v := range []*VkTensorF32{x, w, b, out} {
		expected = append(expected, [3]uint64{uint64(a.state.buffer.buf), v.offset, v.size})
	}
	if !reflect.DeepEqual(expected, ranges) || !reflect.DeepEqual(push, []uint32{5, 2, 3, 3, 2, 0}) || grid != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, grid)
	}
	stage, err := op.Stage(context.Background(), out, x, w, b, 2, VkConvChannelsFirst)
	if err != nil {
		t.Fatal(err)
	}
	stage.PushWords[0] = 9
	stage.Tensors[0] = nil
	again, err := op.Stage(context.Background(), out, x, w, b, 2, VkConvChannelsFirst)
	if err != nil || again.PushWords[0] != 5 || again.Tensors[0] != x {
		t.Fatal("borrowed slices")
	}
	a.Close()
	if _, err := op.Stage(context.Background(), out, x, w, b, 2, VkConvChannelsFirst); err == nil {
		t.Fatal("closed arena")
	}
	op.Close()
}
func TestVulkanOfflineConvAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	k.numBuffers = 4
	k.pushSize = 24
	op := &VkConv1D3F32{kernel: k}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("not dispatch")
	})
	for _, layout := range []VkConvInputLayout{VkConvChannelsFirst, VkConvTimeMajor} {
		for _, stride := range []int{1, 2} {
			x := linearMetadataTensor(1, 2048, 4096)
			if layout == VkConvTimeMajor {
				x = linearMetadataTensor(1, 4096, 2048)
			}
			w, b, out := linearMetadataTensor(2, 2048, 2048, 3), linearMetadataTensor(3, 2048), linearMetadataTensor(4, (4096+stride-1)/stride, 2048)
			s, err := op.Stage(context.Background(), out, x, w, b, stride, layout)
			if err != nil || s.Groups != ([3]uint32{128, uint32((out.shape[0] + 15) / 16), 1}) {
				t.Fatal("maxshape", err)
			}
		}
	}
	x, w, b, out := linearMetadataTensor(1, 2, 5), linearMetadataTensor(2, 3, 2, 3), linearMetadataTensor(3, 3), linearMetadataTensor(4, 3, 3)
	for slot := 0; slot < 4; slot++ {
		for _, mutate := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 0 }, func(v *VkTensorF32) { v.shape[0] = 0 }, func(v *VkTensorF32) { v.size -= 4 }} {
			args := []*VkTensorF32{out, x, w, b}
			bad := *args[slot]
			mutate(&bad)
			args[slot] = &bad
			if _, err := op.Stage(context.Background(), args[0], args[1], args[2], args[3], 2, VkConvChannelsFirst); err == nil {
				t.Fatal("invalidtensor", slot)
			}
		}
	}
	for _, s := range []int{-1, 0, 3} {
		if _, err := op.Stage(context.Background(), out, x, w, b, s, VkConvChannelsFirst); err == nil {
			t.Fatal("stride")
		}
	}
	if _, err := op.Stage(context.Background(), out, x, w, b, 2, 2); err == nil {
		t.Fatal("layout")
	}
	badW := linearMetadataTensor(2, 3, 2, 1)
	if _, err := op.Stage(context.Background(), out, x, badW, b, 2, 0); err == nil {
		t.Fatal("tapshape")
	}
	for _, shape := range [][2]int{{2049, 5}, {2, 4097}} {
		if _, err := op.Stage(context.Background(), out, linearMetadataTensor(1, shape[0], shape[1]), w, b, 2, 0); err == nil {
			t.Fatal("xenvelope")
		}
	}
	for _, src := range []*VkTensorF32{x, w, b} {
		alias := *out
		alias.arena = src.arena
		alias.arena.buffer.size = 256
		if _, err := op.Stage(context.Background(), &alias, x, w, b, 2, 0); err == nil {
			t.Fatal("alias")
		}
	}
	partial := *out
	partial.arena = x.arena
	partial.offset = 16
	if _, err := op.Stage(context.Background(), &partial, x, w, b, 2, 0); err == nil {
		t.Fatal("alignedpartial")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x, w, b, 2, 0)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[0] = 0
	if _, err := op.Stage(context.Background(), out, x, w, b, 2, 0); err == nil {
		t.Fatal("grid")
	}
	vkLimits = offlineLimits()
	vkLimits.StorageBufferRange = 35
	if _, err := op.Stage(context.Background(), out, x, w, b, 2, 0); err == nil {
		t.Fatal("range")
	}
	if len(lane.events) != 0 {
		t.Fatal("native admission", lane.events)
	}
	if _, err := new(VkConv1D3F32).Stage(context.Background(), nil, nil, nil, nil, 1, 0); err == nil {
		t.Fatal("zero")
	}
	if err := (*VkConv1D3F32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkConv1D3F32).Close(); err != nil {
		t.Fatal(err)
	}
}
func TestVulkanOfflineConvContract(t *testing.T) {
	offlineVK(t)
	got, err := InspectVulkanShader(spirv_conv1d3_f32)
	if err != nil || got != (VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 2048, StorageBindings: 15, PushBytes: 24}) {
		t.Fatal(got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkConv1D3F32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.SharedMemoryBytes = 2047
	_, err = NewVkConv1D3F32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	words := spirvTestWords(spirv_conv1d3_f32)
	found := false
	for i := 5; i < len(words); {
		n := int(words[i] >> 16)
		if uint16(words[i]) == 130 {
			found = true
			bad := append([]uint32(nil), words[:i]...)
			bad = append(bad, 4<<16|130)
			bad = append(bad, words[i+1:i+4]...)
			bad = append(bad, words[i+n:]...)
			_, err := InspectVulkanShader(spirvTestBytes(bad))
			expectErrorIs(t, err, ErrVulkanShaderContract)
		}
		i += n
	}
	if !found {
		t.Fatal("missingISub")
	}
}
func TestVulkanOfflineConvAddPlan(t *testing.T) {
	m, k, a, _ := newPlanMock(t)
	a.Close()
	mockAttentionMemory(t)
	a, out, x, w, b := convMockTensors(t)
	k.numBuffers = 4
	k.pushSize = 24
	conv := &VkConv1D3F32{kernel: k}
	addKernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 3, pushSize: 4}
	add := &VkAddF32{kernel: addKernel}
	pos := mustTensor(t, a, 3, 3)
	s1, err := conv.Stage(context.Background(), out, x, w, b, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	s2, err := add.Stage(context.Background(), out, out, pos)
	if err != nil {
		t.Fatal(err)
	}
	var push []uint32
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		push = append(push, n)
	})
	plan := mustPlan(t, s1, s2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	if !reflect.DeepEqual(push, []uint32{24, 4}) {
		t.Fatal(push)
	}
	expectErrorIs(t, conv.Close(), ErrVulkanInFlight)
	expectErrorIs(t, add.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	conv.Close()
	add.Close()
	a.Close()
	assertMemory(t, 0, 0)
}

func TestVulkanOfflineConvCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{16, 16, 1}) || c.SharedBytes != 2048 || c.StorageBindings != 15 || c.PushBytes != 24 {
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
		if *(*uint32)(unsafe.Add(r, 8)) != 24 {
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
	op, err := NewVkConv1D3F32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 4 || op.kernel.pushSize != 24 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
