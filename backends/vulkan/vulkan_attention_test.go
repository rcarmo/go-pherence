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

// Independent full-matrix float64 reference; no tile/online-softmax recurrence.
func attentionReference(q, k, v []float32, m, n, h, d int) []float64 {
	w := h * d
	out := make([]float64, m*w)
	scores := make([]float64, n)
	for r := 0; r < m; r++ {
		for head := 0; head < h; head++ {
			max := -math.MaxFloat64
			for j := 0; j < n; j++ {
				s := 0.
				for c := 0; c < d; c++ {
					s += float64(q[r*w+head*d+c]) * float64(k[j*w+head*d+c])
				}
				scores[j] = s / math.Sqrt(float64(d))
				if scores[j] > max {
					max = scores[j]
				}
			}
			sum := 0.
			for j := range scores {
				scores[j] = math.Exp(scores[j] - max)
				sum += scores[j]
			}
			for c := 0; c < d; c++ {
				s := 0.
				for j := 0; j < n; j++ {
					s += scores[j] / sum * float64(v[j*w+head*d+c])
				}
				out[r*w+head*d+c] = s
			}
		}
	}
	return out
}

// Go model of barrier-separated shader phases. Independent lane/group orders
// exercise shared tile ownership and index/tail rules. No SPIR-V is executed;
// Exp is Go libm rounded to F32, not all device transcendental implementations.
func attentionTileModel(t *testing.T, q, k, v []float32, m, n, h, d int, seed int64) []float32 {
	t.Helper()
	w := h * d
	out := make([]float32, m*w+4)
	for i := m * w; i < len(out); i++ {
		out[i] = 123.25
	}
	writers := make([]int, m*w)
	rng := rand.New(rand.NewSource(seed))
	lanes := rng.Perm(256)
	tiles := (m + 15) / 16
	scale := float32(1 / math.Sqrt(float64(d)))
	for _, g := range rng.Perm(tiles * h) {
		head, queryBase := g/tiles, (g%tiles)*16
		var qt, kv, acc [1024]float32
		var p [256]float32
		var rowMax, rowSum, alpha [16]float32
		load := func(dst *[1024]float32, src []float32, base, rows int) {
			var owners [1024]int
			for _, lane := range lanes {
				for i := lane; i < 1024; i += 256 {
					r, c := i/64, i%64
					dst[i] = 0
					if base+r < rows && c < d {
						dst[i] = src[(base+r)*w+head*d+c]
					}
					owners[i]++
				}
			}
			for _, x := range owners {
				if x != 1 {
					t.Fatal("shared tile writer")
				}
			}
		}
		load(&qt, q, queryBase, m)
		for base := 0; base < n; base += 16 {
			load(&kv, k, base, n)
			for _, lane := range lanes {
				x, y := lane%16, lane/16
				s := float32(0)
				for c := 0; c < d; c++ {
					s = float32(s + float32(qt[y*64+c]*kv[x*64+c]))
				}
				p[lane] = s * scale
			}
			for _, y := range rng.Perm(16) {
				mx := p[y*16]
				for j := 1; j < 16 && base+j < n; j++ {
					if mx < p[y*16+j] {
						mx = p[y*16+j]
					}
				}
				a := float32(0)
				if base > 0 {
					if mx < rowMax[y] {
						mx = rowMax[y]
					}
					a = float32(math.Exp(float64(float32(rowMax[y] - mx))))
				}
				sum := float32(0)
				for j := 0; j < 16; j++ {
					prob := float32(0)
					if base+j < n {
						prob = float32(math.Exp(float64(float32(p[y*16+j] - mx))))
					}
					p[y*16+j] = prob
					sum += prob
				}
				rowSum[y] = float32(float32(rowSum[y]*a) + sum)
				rowMax[y] = mx
				alpha[y] = a
			}
			load(&kv, v, base, n)
			for _, lane := range lanes {
				x, y := lane%16, lane/16
				for i := 0; i < 4; i++ {
					c := x + i*16
					idx := y*64 + c
					acc[idx] *= alpha[y]
					if c < d {
						for j := 0; j < 16; j++ {
							acc[idx] = float32(acc[idx] + float32(p[y*16+j]*kv[j*64+c]))
						}
					}
				}
			}
		}
		for _, lane := range lanes {
			x, y := lane%16, lane/16
			if queryBase+y < m {
				if rowSum[y] <= 0 || math.IsNaN(float64(rowSum[y])) {
					t.Fatal("invalid denominator")
				}
				for i := 0; i < 4; i++ {
					c := x + i*16
					if c < d {
						idx := (queryBase+y)*w + head*d + c
						out[idx] = acc[y*64+c] / rowSum[y]
						writers[idx]++
					}
				}
			}
		}
	}
	for _, n := range writers {
		if n != 1 {
			t.Fatal("output ownership", n)
		}
	}
	for _, x := range out[m*w:] {
		if x != 123.25 {
			t.Fatal("tail canary")
		}
	}
	return out[:m*w]
}
func TestVulkanOfflineAttentionNumerics(t *testing.T) {
	dims := [][4]int{{1, 1, 1, 1}, {2, 3, 2, 3}, {15, 17, 3, 7}, {16, 16, 2, 16}, {17, 31, 2, 17}, {31, 33, 3, 31}, {33, 17, 2, 32}, {3, 65, 2, 63}, {2, 129, 3, 64}, {1, 4096, 1, 1}, {4096, 1, 1, 1}, {1, 17, 32, 64}}
	samples := 0
	maxAbs := 0.
	for _, s := range dims {
		m, n, h, d := s[0], s[1], s[2], s[3]
		rng := rand.New(rand.NewSource(int64(m + n + h + d)))
		q, k, v := make([]float32, m*h*d), make([]float32, n*h*d), make([]float32, n*h*d)
		for _, a := range [][]float32{q, k, v} {
			for i := range a {
				a[i] = float32(rng.Float64()*4 - 2)
			}
		}
		ref := attentionReference(q, k, v, m, n, h, d)
		var first []float32
		for seed := int64(0); seed < 4; seed++ {
			out := attentionTileModel(t, q, k, v, m, n, h, d, seed)
			if seed == 0 {
				first = out
			} else if !reflect.DeepEqual(first, out) {
				t.Fatal("lane schedule changed result", s)
			}
			for i, x := range out {
				e := math.Abs(float64(x) - ref[i])
				maxAbs = math.Max(maxAbs, e)
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || e > 2e-5+2e-5*math.Abs(ref[i]) {
					t.Fatal("oracle", s, i, x, ref[i], e)
				}
				samples++
			}
		}
	}
	t.Logf("shapes=%d permutations=4 compared=%d max_abs=%g tolerance=2e-5+2e-5*abs(ref)", len(dims), samples, maxAbs)
}
func TestVulkanOfflineAttentionSoftmaxStress(t *testing.T) {
	// Tile maxima rise, fall, tie and differ by >1000. All logits remain finite.
	// Larger absolute budget isolates F32 logit cancellation at large magnitudes;
	// this source-model stress budget is separate from ordinary bounded fixtures.
	maxAbs, compared := 0.0, 0
	for _, sign := range []float32{-1, 0, 1} {
		m, n, h, d := 17, 49, 2, 3
		w := h * d
		q, k, v := make([]float32, m*w), make([]float32, n*w), make([]float32, n*w)
		for i := range q {
			q[i] = sign
		}
		for j := 0; j < n; j++ {
			for c := 0; c < w; c++ {
				k[j*w+c] = []float32{-800, 900, 900, -1000}[j/16] + float32(j%3)
				v[j*w+c] = float32(j%7-3) * float32(c+1)
			}
		}
		ref := attentionReference(q, k, v, m, n, h, d)
		out := attentionTileModel(t, q, k, v, m, n, h, d, 15)
		for i, x := range out {
			maxAbs = math.Max(maxAbs, math.Abs(float64(x)-ref[i]))
			compared++
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x)-ref[i]) > 2e-4+2e-5*math.Abs(ref[i]) {
				t.Fatal("rescale", sign, i, x, ref[i])
			}
		}
	}
	t.Logf("stress_compared=%d max_abs=%g tolerance=2e-4+2e-5*abs(ref)", compared, maxAbs)
	// Single key: output equals that head's value, independent of query/logit.
	out := attentionTileModel(t, []float32{2, 3, -1, 4}, []float32{4, -7}, []float32{11, -13}, 2, 1, 1, 2, 7)
	if !reflect.DeepEqual(out, []float32{11, -13, 11, -13}) {
		t.Fatal("singlekey", out)
	}
}

// Larger bounded Go-backed mock allocations; no loader or device access.
func mockAttentionMemory(t *testing.T) {
	t.Helper()
	m := mockMemory(t)
	m.requirement = 2048
	storage := map[VkDeviceMemory]*[2048]byte{}
	mockVK(t, &vkMapMemory, func(d VkDevice, h VkDeviceMemory, o, n uint64, flags uint32, out *unsafe.Pointer) VkResult {
		if o != 0 || n > 2048 {
			t.Fatal("attention mock map extent")
		}
		data := new([2048]byte)
		storage[h] = data
		*out = unsafe.Pointer(data)
		return VK_SUCCESS
	})
	mockVK(t, &vkFreeMemory, func(d VkDevice, h VkDeviceMemory, p unsafe.Pointer) {
		m.frees++
		delete(m.storage, h)
		delete(storage, h)
	})
}
func attentionMockTensors(t *testing.T) (*VkTensorArena, *VkTensorF32, *VkTensorF32, *VkTensorF32, *VkTensorF32) {
	a := mustArena(t, 2048)
	return a, mustTensor(t, a, 17, 6), mustTensor(t, a, 17, 6), mustTensor(t, a, 3, 6), mustTensor(t, a, 3, 6)
}
func TestVulkanOfflineAttentionBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	kernel.numBuffers = 4
	kernel.pushSize = 20
	op := &VkAttentionF32{kernel: kernel}
	a, out, q, k, v := attentionMockTensors(t)
	var ranges [][3]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, z unsafe.Pointer) {
		ranges = nil
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if s != 0x20 || off != 0 || n != 20 {
			t.Fatal("pushheader")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 5)...)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, q, k, v, 2); err != nil {
		t.Fatal(err)
	}
	expected := [][3]uint64{}
	for _, x := range []*VkTensorF32{q, k, v, out} {
		expected = append(expected, [3]uint64{uint64(a.state.buffer.buf), x.offset, x.size})
	}
	if !reflect.DeepEqual(ranges, expected) || groups != ([3]uint32{2, 2, 1}) || !reflect.DeepEqual(push, []uint32{17, 3, 2, 3, math.Float32bits(float32(1 / math.Sqrt(3)))}) {
		t.Fatal(ranges, push, groups)
	}
	s, err := op.Stage(context.Background(), out, q, k, v, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Tensors[0] = nil
	s.PushWords[0] = 9
	again, err := op.Stage(context.Background(), out, q, k, v, 2)
	if err != nil || again.Tensors[0] != q || again.PushWords[0] != 17 {
		t.Fatal("borrowed stage")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := op.Stage(context.Background(), out, q, k, v, 2); err == nil {
		t.Fatal("closed arena")
	}
	op.Close()
}
func TestVulkanOfflineAttentionAdmission(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 4
	kernel.pushSize = 20
	op := &VkAttentionF32{kernel: kernel}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("no dispatch")
	})
	for _, s := range [][4]int{{1, 1, 1, 1}, {4096, 4096, 32, 64}, {17, 19, 20, 64}} {
		m, n, h, d := s[0], s[1], s[2], s[3]
		q, k, v, out := linearMetadataTensor(1, m, h*d), linearMetadataTensor(2, n, h*d), linearMetadataTensor(3, n, h*d), linearMetadataTensor(4, m, h*d)
		stage, err := op.Stage(context.Background(), out, q, k, v, h)
		if err != nil || stage.Groups != ([3]uint32{uint32((m + 15) / 16), uint32(h), 1}) {
			t.Fatal(s, err)
		}
	}
	q, k, v, out := linearMetadataTensor(1, 2, 6), linearMetadataTensor(2, 3, 6), linearMetadataTensor(3, 3, 6), linearMetadataTensor(4, 2, 6)
	for slot := 0; slot < 4; slot++ {
		for _, mutate := range []func(*VkTensorF32){func(x *VkTensorF32) { x.rank = 1 }, func(x *VkTensorF32) { x.shape[0] = 0 }, func(x *VkTensorF32) { x.shape[0] = 4097 }, func(x *VkTensorF32) { x.shape[1] = 7 }, func(x *VkTensorF32) { x.size -= 4 }} {
			args := []*VkTensorF32{out, q, k, v}
			bad := *args[slot]
			mutate(&bad)
			args[slot] = &bad
			if _, err := op.Stage(context.Background(), args[0], args[1], args[2], args[3], 2); err == nil {
				t.Fatal("bad tensor", slot)
			}
		}
	}
	for _, heads := range []int{-1, 0, 4, 33} {
		if _, err := op.Stage(context.Background(), out, q, k, v, heads); err == nil {
			t.Fatal("bad heads", heads)
		}
	}
	for _, width := range []int{65, 2049} {
		x, y := linearMetadataTensor(1, 2, width), linearMetadataTensor(4, 2, width)
		if _, err := op.Stage(context.Background(), y, x, x, x, 1); err == nil {
			t.Fatal("head width")
		}
	}
	for _, alias := range []*VkTensorF32{q, k, v} {
		x := *alias
		x.shape[0] = 2
		x.size = 48
		if _, err := op.Stage(context.Background(), &x, q, k, v, 2); err == nil {
			t.Fatal("output alias")
		}
	}
	partial := *q
	partial.offset = 16
	q.arena.buffer.size = 64
	if _, err := op.Stage(context.Background(), &partial, q, k, v, 2); err == nil {
		t.Fatal("aligned partial alias")
	}
	if _, err := op.Stage(context.Background(), out, q, q, q, 2); err != nil {
		t.Fatal("read input alias", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, q, k, v, 2)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[1] = 1
	if _, err := op.Stage(context.Background(), out, q, k, v, 2); err == nil {
		t.Fatal("head grid")
	}
	vkLimits = offlineLimits()
	vkLimits.StorageBufferRange = 47
	if _, err := op.Stage(context.Background(), out, q, k, v, 2); err == nil {
		t.Fatal("descriptor range")
	}
	vkLimits = offlineLimits()
	if len(lane.events) != 0 {
		t.Fatal("native admission", lane.events)
	}
	if _, err := op.Stage(context.Background(), nil, q, k, v, 2); err == nil {
		t.Fatal("nil output")
	}
	if _, err := op.Stage(context.Background(), out, nil, k, v, 2); err == nil {
		t.Fatal("nil Q")
	}
	if err := (*VkAttentionF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkAttentionF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkAttentionF32).Stage(context.Background(), nil, nil, nil, nil, 1); err == nil {
		t.Fatal("zero operator")
	}
}
func TestVulkanOfflineAttentionPlan(t *testing.T) {
	m, k, a, _ := newPlanMock(t)
	a.Close()
	mockAttentionMemory(t)
	a, out, q, key, v := attentionMockTensors(t)
	k.numBuffers = 4
	k.pushSize = 20
	attention := &VkAttentionF32{kernel: k}
	projKernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 4, pushSize: 12}
	linear := &VkLinearF32{kernel: projKernel}
	params := mustArena(t, 512)
	w := mustTensor(t, params, 6, 6)
	b := mustTensor(t, params, 6)
	first, err := attention.Stage(context.Background(), out, q, key, v, 2)
	if err != nil {
		t.Fatal(err)
	}
	second, err := linear.Stage(context.Background(), q, out, w, b)
	if err != nil {
		t.Fatal(err)
	}
	var push []uint32
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		push = append(push, n)
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
	if !reflect.DeepEqual(push, []uint32{20, 12}) {
		t.Fatal("push", push)
	}
	if vkPending == nil || len(vkPending.buffers) != 2 {
		t.Fatal("retention")
	}
	expectErrorIs(t, attention.Close(), ErrVulkanInFlight)
	expectErrorIs(t, linear.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	expectErrorIs(t, params.Close(), ErrVulkanInFlight)
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	attention.Close()
	linear.Close()
	a.Close()
	params.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineAttentionContract(t *testing.T) {
	offlineVK(t)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 9408, StorageBindings: 15, PushBytes: 20}
	got, err := InspectVulkanShader(spirv_attention_f32)
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkAttentionF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.SharedMemoryBytes = 9407
	_, err = NewVkAttentionF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	vkLimits = offlineLimits()
	vkLimits.WorkgroupSize[1] = 15
	_, err = NewVkAttentionF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}

func TestVulkanOfflineAttentionCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{16, 16, 1}) || c.SharedBytes != 9408 || c.StorageBindings != 15 || c.PushBytes != 20 {
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
		if *(*uint32)(unsafe.Add(r, 8)) != 20 {
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
	op, err := NewVkAttentionF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 4 || op.kernel.pushSize != 20 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
