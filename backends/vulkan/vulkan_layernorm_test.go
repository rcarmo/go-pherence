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

// Shader schedule model, NOT SPIR-V execution. The oracle below instead uses
// serial float64 centered population variance, as Whisper's scalar oracle does.
func layerNormSchedule(x, w, b []float32, rows, width int, eps float32, inplace bool, seed int64) []float32 {
	data := append([]float32(nil), x...)
	out := make([]float32, len(x))
	if inplace {
		out = data
	}
	random := rand.New(rand.NewSource(seed))
	for _, row := range random.Perm(rows) {
		base := row * width
		var shared [256]float32
		for _, tid := range random.Perm(256) {
			for col := tid; col < width; col += 256 {
				shared[tid] += data[base+col]
			}
		}
		for stride := 128; stride > 0; stride /= 2 {
			for _, tid := range random.Perm(stride) {
				shared[tid] += shared[tid+stride]
			}
		}
		mean := shared[0] / float32(width)
		clear(shared[:])
		for _, tid := range random.Perm(256) {
			for col := tid; col < width; col += 256 {
				d := data[base+col] - mean
				shared[tid] += d * d
			}
		}
		for stride := 128; stride > 0; stride /= 2 {
			for _, tid := range random.Perm(stride) {
				shared[tid] += shared[tid+stride]
			}
		}
		inv := float32(1 / math.Sqrt(float64(shared[0]/float32(width)+eps)))
		for _, tid := range random.Perm(256) {
			for col := tid; col < width; col += 256 {
				out[base+col] = ((data[base+col]-mean)*inv)*w[col] + b[col]
			}
		}
	}
	return out
}
func TestVulkanOfflineLayerNormSchedule(t *testing.T) {
	for _, width := range []int{1, 2, 3, 31, 255, 256, 257, 384, 768, 1280, 16384} {
		const rows = 3
		eps := float32(1e-5)
		x := make([]float32, rows*width)
		w := make([]float32, width)
		b := make([]float32, width)
		for col := range w {
			w[col] = float32(col%11-5) / 7
			b[col] = float32(col%7-3) / 9
		}
		for i := range x {
			x[i] = float32(i%41-20) / 13
			if i/width == 1 {
				x[i] = 3.25
			}
			if i/width == 2 {
				x[i] = float32(i%5) * 0.0001
			}
		}
		oracle := make([]float64, len(x))
		for row := 0; row < rows; row++ {
			var mean, variance float64
			for _, v := range x[row*width : (row+1)*width] {
				mean += float64(v)
			}
			mean /= float64(width)
			for _, v := range x[row*width : (row+1)*width] {
				d := float64(v) - mean
				variance += d * d
			}
			inv := 1 / math.Sqrt(variance/float64(width)+float64(eps))
			for col := 0; col < width; col++ {
				oracle[row*width+col] = (float64(x[row*width+col])-mean)*inv*float64(w[col]) + float64(b[col])
			}
		}
		for seed := int64(0); seed < 4; seed++ {
			out := layerNormSchedule(x, w, b, rows, width, eps, false, seed)
			alias := layerNormSchedule(x, w, b, rows, width, eps, true, seed)
			if !reflect.DeepEqual(out, alias) {
				t.Fatal("exact alias schedule differs", width, seed)
			}
			for i, v := range out {
				if math.IsNaN(float64(v)) || math.Abs(float64(v)-oracle[i]) > 2e-5+2e-5*math.Abs(oracle[i]) {
					t.Fatalf("width%d seed%d index%d got%.9g want%.9g", width, seed, i, v, oracle[i])
				}
			}
		}
	}
}
func layerNormMockTensors(t *testing.T) (*VkTensorArena, *VkTensorF32, *VkTensorF32, *VkTensorF32, *VkTensorF32) {
	t.Helper()
	a := mustArena(t, 64)
	x := mustTensor(t, a, 2, 2)
	w := mustTensor(t, a, 2)
	b := mustTensor(t, a, 2)
	out := mustTensor(t, a, 2, 2)
	return a, out, x, w, b
}
func TestVulkanOfflineLayerNormBindings(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockMemory(t)
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLayerNormF32{kernel: k}
	a, out, x, w, b := layerNormMockTensors(t)
	var ranges [][3]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		ranges = nil
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, stage, off, n uint32, p unsafe.Pointer) {
		if n != 12 || off != 0 || stage != 0x20 {
			t.Fatal("push ABI")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 3)...)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, w, b, 1e-5); err != nil {
		t.Fatal(err)
	}
	h := uint64(a.state.buffer.buf)
	if !reflect.DeepEqual(ranges, [][3]uint64{{h, 0, 16}, {h, 16, 8}, {h, 32, 8}, {h, 48, 16}}) || !reflect.DeepEqual(push, []uint32{2, 2, math.Float32bits(1e-5)}) || groups != ([3]uint32{2, 1, 1}) {
		t.Fatal("operator binding", ranges, push, groups)
	}
	if err := op.Forward(context.Background(), x, x, w, b, 1e-5); err != nil {
		t.Fatal("exact alias", err)
	}
	if ranges[0] != ranges[3] {
		t.Fatal("not exact alias")
	}
	stage, err := op.Stage(context.Background(), out, x, w, b, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	stage.PushWords[0] = 99
	stage.Tensors[0] = nil
	second, err := op.Stage(context.Background(), out, x, w, b, 1e-5)
	if err != nil || second.PushWords[0] != 2 || second.Tensors[0] != x {
		t.Fatal("borrowed stage")
	}
	op.Close()
	expectErrorIs(t, op.Forward(context.Background(), out, x, w, b, 1e-5), ErrVulkanClosed)
	a.Close()
}
func TestVulkanOfflineLayerNormAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	mockMemory(t)
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLayerNormF32{kernel: k}
	a, out, x, w, b := layerNormMockTensors(t)
	for _, eps := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := op.Stage(context.Background(), out, x, w, b, eps); err == nil {
			t.Fatal("bad epsilon")
		}
	}
	for _, change := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 1 }, func(v *VkTensorF32) { v.shape[1] = 3 }, func(v *VkTensorF32) { v.shape[1] = 16385 }, func(v *VkTensorF32) { v.shape[0] = int(^uint(0) >> 1) }, func(v *VkTensorF32) { v.size = 12 }} {
		v := *x
		change(&v)
		if _, err := op.Stage(context.Background(), out, &v, w, b, 1e-5); err == nil {
			t.Fatal("invalid shape")
		}
	}
	badWeight := *w
	badWeight.rank = 2
	if _, err := op.Stage(context.Background(), out, x, &badWeight, b, 1e-5); err == nil {
		t.Fatal("weight rank")
	}
	// Same allocation, shifted range: reject before any recording. Adjust size
	///shape consistently to test overlap rather than a geometry-only failure.
	shifted := *x
	shifted.offset = 4
	if _, err := op.Stage(context.Background(), &shifted, x, w, b, 1e-5); err == nil {
		t.Fatal("partial alias")
	}
	paramAlias := *w
	paramAlias.offset = 48
	if _, err := op.Stage(context.Background(), out, x, &paramAlias, b, 1e-5); err == nil {
		t.Fatal("output clobbers weights")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x, w, b, 1e-5)
	expectErrorIs(t, err, context.Canceled)
	if len(lane.events) != 0 {
		t.Fatal("invalid request recorded")
	}
	a.Close()
}
func TestVulkanOfflineLayerNormPlan(t *testing.T) {
	m, k, a, x := newPlanMock(t)
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLayerNormF32{kernel: k}
	// Original arena has1scalar reserved; use a new arena for matching rank2 data.
	a.Close()
	a, out, x, w, b := layerNormMockTensors(t)
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, stage, off, n uint32, p unsafe.Pointer) {
		if c != 500 || n != 12 {
			t.Fatal("plan push")
		}
	})
	first, err := op.Stage(context.Background(), out, x, w, b, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	second, err := op.Stage(context.Background(), out, out, w, b, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	p := mustPlan(t, first, second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, p.Run(ctx), ErrVulkanInFlight)
	if vkPending == nil || len(vkPending.buffers) != 1 {
		t.Fatal("operator plan retention")
	}
	expectErrorIs(t, op.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	p.Close()
	op.Close()
	a.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineLayerNormConstructor(t *testing.T) {
	offlineVK(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewVkLayerNormF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.SharedMemoryBytes = 1023
	_, err = NewVkLayerNormF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	vkLimits = offlineLimits()
	vkLimits.PerStageStorageBuffers = 3
	_, err = NewVkLayerNormF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	c, err := InspectVulkanShader(spirv_layer_norm_f32)
	if err != nil || c != (VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, SharedBytes: 1024, StorageBindings: 15, PushBytes: 12}) {
		t.Fatal(c, err)
	}
}

func TestVulkanOfflineLayerNormZeroValue(t *testing.T) {
	// Delegate flagged a possible nil-kernel panic, but VkComputeKernel.Close
	// already accepts nil receivers. Lock that transitive contract down.
	if err := (*VkLayerNormF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkLayerNormF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkLayerNormF32).Stage(context.Background(), nil, nil, nil, nil, 1e-5); err == nil {
		t.Fatal("zero-value stage accepted")
	}
}

func TestVulkanOfflineLayerNormCreationSuccess(t *testing.T) {
	offlineVK(t)
	_, k, _ := newLifetimeMock(t)
	_ = k
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		contract, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || contract.StorageBindings != 15 || contract.PushBytes != 12 || contract.SharedBytes != 1024 {
			t.Fatal("constructor shader", contract, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(p, 20)) != 4 {
			t.Fatal("descriptor count")
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
	destroyed := 0
	mockVK(t, &vkDestroyShaderModule, func(VkDevice, VkShaderModule, unsafe.Pointer) { destroyed++ })
	op, err := NewVkLayerNormF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 || destroyed != 1 {
		t.Fatal("constructor binding")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
