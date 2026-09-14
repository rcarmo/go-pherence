package vulkan

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

// Model shared tile loading/consumption with shuffled lanes; explicit phases
// stand for workgroup barriers. No SPIR-V/GPU execution or speed measurement.
func linearTileModel(x, w, b []float32, m, k, n int, seed int64) ([]float32, []int) {
	out := make([]float32, m*n)
	writers := make([]int, m*n)
	r := rand.New(rand.NewSource(seed))
	gridX, gridY := (n+15)/16, (m+15)/16
	for _, tile := range r.Perm(gridX * gridY) {
		gx, gy := tile%gridX, tile/gridX
		var sums, tx, tw [256]float32
		for base := 0; base < k; base += 16 {
			for _, lane := range r.Perm(256) {
				lx, ly := lane%16, lane/16
				row, wr := gy*16+ly, gx*16+ly
				tx[lane] = 0
				tw[lane] = 0
				if base+lx < k {
					if row < m {
						tx[lane] = x[row*k+base+lx]
					}
					if wr < n {
						tw[lane] = w[wr*k+base+lx]
					}
				}
			}
			for _, lane := range r.Perm(256) {
				lx, ly := lane%16, lane/16
				for q := 0; q < 16; q++ {
					sums[lane] += tx[ly*16+q] * tw[lx*16+q]
				}
			}
		}
		for _, lane := range r.Perm(256) {
			row, col := gy*16+lane/16, gx*16+lane%16
			if row < m && col < n {
				out[row*n+col] = sums[lane] + b[col]
				writers[row*n+col]++
			}
		}
	}
	return out, writers
}
func TestVulkanOfflineLinearTileSchedule(t *testing.T) {
	for _, dims := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {16, 17, 15}, {17, 31, 19}, {31, 33, 7}, {33, 17, 31}, {3, 384, 9}, {2, 1280, 7}, {1, 16384, 2}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, w, b := make([]float32, m*k), make([]float32, n*k), make([]float32, n)
		for i := range x {
			x[i] = float32(i%29-14) / 17
		}
		for i := range w {
			w[i] = float32((i*11)%31-15) / 23
		}
		for i := range b {
			b[i] = float32(i%7-3) / 5
		}
		oracle := make([]float64, m*n)
		serial := make([]float32, m*n)
		for row := 0; row < m; row++ {
			for col := 0; col < n; col++ {
				for q := 0; q < k; q++ {
					oracle[row*n+col] += float64(x[row*k+q]) * float64(w[col*k+q])
					serial[row*n+col] += x[row*k+q] * w[col*k+q]
				}
				oracle[row*n+col] += float64(b[col])
				serial[row*n+col] += b[col]
			}
		}
		for seed := int64(0); seed < 4; seed++ {
			got, writers := linearTileModel(x, w, b, m, k, n, seed)
			for i, v := range got {
				if writers[i] != 1 {
					t.Fatal("output owner count", dims, i, writers[i])
				}
				if math.Float32bits(v) != math.Float32bits(serial[i]) {
					t.Fatal("tile differs from serial schedule", dims, i)
				}
				if math.IsNaN(float64(v)) || math.Abs(float64(v)-oracle[i]) > 2e-5+2e-5*math.Abs(oracle[i]) {
					t.Fatalf("shape%v seed%d output%d got%.9g want%.9g", dims, seed, i, v, oracle[i])
				}
			}
		}
	}
}
func linearMockTensors(t *testing.T) (*VkTensorArena, *VkTensorF32, *VkTensorF32, *VkTensorF32, *VkTensorF32) {
	t.Helper()
	a := mustArena(t, 64)
	x := mustTensor(t, a, 2, 2)
	w := mustTensor(t, a, 2, 2)
	b := mustTensor(t, a, 2)
	out := mustTensor(t, a, 2, 2)
	return a, out, x, w, b
}
func TestVulkanOfflineLinearBindings(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockMemory(t)
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLinearF32{kernel: k}
	a, out, x, w, b := linearMockTensors(t)
	var ranges [][3]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if s != 0x20 || off != 0 || n != 12 {
			t.Fatal("push header")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 3)...)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, w, b); err != nil {
		t.Fatal(err)
	}
	h := uint64(a.state.buffer.buf)
	if !reflect.DeepEqual(ranges, [][3]uint64{{h, 0, 16}, {h, 16, 16}, {h, 32, 8}, {h, 48, 16}}) || !reflect.DeepEqual(push, []uint32{2, 2, 2}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	first, err := op.Stage(context.Background(), out, x, w, b)
	if err != nil {
		t.Fatal(err)
	}
	first.PushWords[0] = 0
	first.Tensors[0] = nil
	again, err := op.Stage(context.Background(), out, x, w, b)
	if err != nil || again.PushWords[0] != 2 || again.Tensors[0] != x {
		t.Fatal("stage data not independent")
	}
	a.Close()
	if _, err := op.Stage(context.Background(), out, x, w, b); err == nil {
		t.Fatal("closed arena admitted")
	}
	op.Close()
}

// Metadata-only tensors for checking large grid/index bounds without allocation
// or mapped access. Only the real arena constructor can create public views.
func linearMetadataTensor(id int, shape ...int) *VkTensorF32 {
	bytes, err := vkF32ShapeBytes(shape)
	if err != nil {
		panic(err)
	}
	back := new(float32)
	t := &VkTensorF32{arena: &vkTensorArenaState{buffer: &VkBuf{device: 100, buf: VkBuffer(id), mem: VkDeviceMemory(id + 100), size: bytes, mapped: unsafe.Pointer(back)}}, size: bytes, rank: len(shape)}
	copy(t.shape[:], shape)
	return t
}
func TestVulkanOfflineLinearGeometryAndAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("admission must not push")
	})
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLinearF32{kernel: k}
	for _, dims := range [][3]int{{1, 1, 1}, {15, 17, 16}, {16, 17, 17}, {17, 15, 31}, {33, 2, 17}, {16384, 1, 1}} {
		m, in, n := dims[0], dims[1], dims[2]
		out, x, w, b := linearMetadataTensor(1, m, n), linearMetadataTensor(2, m, in), linearMetadataTensor(3, n, in), linearMetadataTensor(4, n)
		s, err := op.Stage(context.Background(), out, x, w, b)
		if err != nil || s.Groups != ([3]uint32{uint32((n + 15) / 16), uint32((m + 15) / 16), 1}) || !reflect.DeepEqual(s.PushWords, []uint32{uint32(m), uint32(in), uint32(n)}) {
			t.Fatal(dims, s, err)
		}
	}
	out, x, w, b := linearMetadataTensor(1, 2, 2), linearMetadataTensor(2, 2, 2), linearMetadataTensor(3, 2, 2), linearMetadataTensor(4, 2)
	for _, mutate := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 1 }, func(v *VkTensorF32) { v.shape[0] = 0 }, func(v *VkTensorF32) { v.shape[0] = 16385 }, func(v *VkTensorF32) { v.shape[1] = 16385 }, func(v *VkTensorF32) { v.size = 12 }} {
		v := *x
		mutate(&v)
		if _, err := op.Stage(context.Background(), out, &v, w, b); err == nil {
			t.Fatal("invalid X shape")
		}
	}
	for _, v := range []*VkTensorF32{linearMetadataTensor(5, 2, 3), linearMetadataTensor(5, 3, 2), linearMetadataTensor(5, 2)} {
		if _, err := op.Stage(context.Background(), out, x, v, b); err == nil {
			t.Fatal("invalid W shape")
		}
	}
	if _, err := op.Stage(context.Background(), out, x, w, linearMetadataTensor(5, 3)); err == nil {
		t.Fatal("bias mismatch")
	}
	if _, err := op.Stage(context.Background(), x, x, w, b); err == nil {
		t.Fatal("exact input alias")
	}
	if _, err := op.Stage(context.Background(), w, x, w, b); err == nil {
		t.Fatal("output overlaps weights")
	}
	// Same backing buffer with nonzero shifted overlap.
	partial := *out
	partial.arena = x.arena
	partial.offset = 4
	x.arena.buffer.size = 32
	if _, err := op.Stage(context.Background(), &partial, x, w, b); err == nil {
		t.Fatal("partial overlap")
	}
	// Read-only input alias remains legal, if their shapes agree.
	if _, err := op.Stage(context.Background(), out, x, x, b); err != nil {
		t.Fatal("read alias rejected", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x, w, b)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[1] = 1
	if _, err := op.Stage(context.Background(), linearMetadataTensor(6, 17, 2), linearMetadataTensor(7, 17, 2), w, b); err == nil {
		t.Fatal("row grid exceeds device")
	}
	if len(lane.events) != 0 {
		t.Fatal("admission invoked native", lane.events)
	}
}
func TestVulkanOfflineLinearLayerNormPlan(t *testing.T) {
	m, k, a, _ := newPlanMock(t)
	a.Close()
	k.numBuffers = 4
	k.pushSize = 12
	mockMemory(t)
	a, out, x, w, b := linearMockTensors(t)
	linear := &VkLinearF32{kernel: k}
	lnKernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 4, pushSize: 12}
	ln := &VkLayerNormF32{kernel: lnKernel}
	params := mustArena(t, 32)
	scale := mustTensor(t, params, 2)
	bias := mustTensor(t, params, 2)
	first, err := linear.Stage(context.Background(), out, x, w, b)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ln.Stage(context.Background(), out, out, scale, bias, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if c != 500 || n != 12 {
			t.Fatal("push ABI")
		}
	})
	plan := mustPlan(t, first, second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	if vkPending == nil || len(vkPending.buffers) != 2 {
		t.Fatal("plan backing retention")
	}
	expectErrorIs(t, linear.Close(), ErrVulkanInFlight)
	expectErrorIs(t, ln.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	linear.Close()
	ln.Close()
	a.Close()
	params.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineLinearConstructor(t *testing.T) {
	offlineVK(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewVkLinearF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.SharedMemoryBytes = 2047
	_, err = NewVkLinearF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	vkLimits = offlineLimits()
	vkLimits.WorkgroupSize[1] = 15
	_, err = NewVkLinearF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	c, err := InspectVulkanShader(spirv_linear_f32)
	if err != nil || c != (VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 2048, StorageBindings: 15, PushBytes: 12}) {
		t.Fatal(c, err)
	}
	if err := (*VkLinearF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkLinearF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkLinearF32).Stage(context.Background(), nil, nil, nil, nil); err == nil {
		t.Fatal("zero value stage")
	}
}

func TestVulkanOfflineLinearTinyAnalytic(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	w := []float32{1, 0, 0, 0, 1, 1}
	b := []float32{10, -1}
	got, _ := linearTileModel(x, w, b, 2, 3, 2, 4)
	if !reflect.DeepEqual(got, []float32{11, 4, 14, 10}) {
		t.Fatal(fmt.Sprint(got))
	}
}

func TestVulkanOfflineLinearCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{16, 16, 1}) || c.SharedBytes != 2048 || c.StorageBindings != 15 || c.PushBytes != 12 {
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
	op, err := NewVkLinearF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
