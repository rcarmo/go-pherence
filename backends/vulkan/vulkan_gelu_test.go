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

// Explicit float32 evaluation matching shader operation order. Exp is rounded
// from Go float64 libm, NOT a model of every Vulkan driver's transcendental.
func geluErfModel(v float32) float32 {
	if v >= 10 {
		return v
	}
	if v <= -10 {
		return 0
	}
	a := float32(math.Abs(float64(v))) * float32(0.7071067811865475244)
	t := float32(1) / float32(float32(1)+float32(float32(0.3275911)*a))
	p := float32(1.061405429)
	p = float32(p*t) - float32(1.453152027)
	p = float32(p*t) + float32(1.421413741)
	p = float32(p*t) - float32(0.284496736)
	p = float32(p*t) + float32(0.254829592)
	e := float32(math.Exp(float64(float32(-a * a))))
	tail := float32(p*t) * e
	if v < 0 {
		return float32(float32(0.5)*v) * tail
	}
	return v * float32(float32(1)-float32(float32(0.5)*tail))
}
func TestVulkanOfflineGELUErfApproximation(t *testing.T) {
	values := make([]float32, 0, 140010)
	for i := -60000; i <= 60000; i++ {
		values = append(values, float32(i)/6000)
	}
	random := rand.New(rand.NewSource(19))
	for i := 0; i < 20000; i++ {
		v := math.Float32frombits(random.Uint32())
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			values = append(values, v)
		}
	}
	for _, v := range []float32{0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, -math.MaxFloat32, 10, -10, math.Nextafter32(10, 0), math.Nextafter32(-10, 0), math.Nextafter32(10, 11), math.Nextafter32(-10, -11)} {
		values = append(values, v)
	}
	var maxAbs, maxRelative, maxCPU float64
	var worst float32
	for _, v := range values {
		// Stable full-erfc oracle avoids libm1+erf cancellation in negative tails.
		ref := 0.5 * float64(v) * math.Erfc(-float64(v)/math.Sqrt2)
		got := float64(geluErfModel(v))
		err := math.Abs(got - ref)
		if math.IsNaN(got) || math.IsInf(got, 0) || err > 2e-6+2e-6*math.Abs(ref) {
			t.Fatalf("v%.9g got%.12g ref%.12g err%.4g", v, got, ref, err)
		}
		if err > maxAbs {
			maxAbs = err
			worst = v
		}
		if math.Abs(ref) > 1e-5 {
			maxRelative = math.Max(maxRelative, err/math.Abs(ref))
		}
		var cpuOut [1]float32
		simd.GELUExact(cpuOut[:], []float32{v})
		cpu := float64(cpuOut[0])
		cpuErr := math.Abs(got - cpu)
		maxCPU = math.Max(maxCPU, cpuErr)
		if cpuErr > 2e-6+2e-6*math.Abs(cpu) {
			t.Fatal("Whisper CPU exact GELU comparison", v, got, cpu)
		}
	}
	// The tanh variant would exceed this test's budget at x=2.
	v := float32(2)
	if math.Abs(float64(geluErfModel(v)-simd.GELUTanhScalar(v))) < 1e-5 {
		t.Fatal("accidentally testing tanh approximation")
	}
	t.Logf("samples=%d max_abs=%g worst_x=%g max_relative_above1e-5=%g max_cpu_difference=%g", len(values), maxAbs, worst, maxRelative, maxCPU)
}
func TestVulkanOfflineGELUErfTailSchedule(t *testing.T) {
	for _, n := range []int{1, 3, 255, 256, 257, 511, 513} {
		input := make([]float32, n+4)
		for i := range input {
			input[i] = float32(i%61-30) / 7
		}
		want := append([]float32(nil), input...)
		for i := 0; i < n; i++ {
			want[i] = geluErfModel(input[i])
		}
		for seed := int64(0); seed < 4; seed++ {
			out := append([]float32(nil), input...)
			writers := make([]int, n)
			for _, i := range rand.New(rand.NewSource(seed)).Perm(((n + 255) / 256) * 256) {
				if i < n {
					out[i] = geluErfModel(out[i])
					writers[i]++
				}
			}
			if !reflect.DeepEqual(out, want) {
				t.Fatal("alias/tail mismatch", n, seed)
			}
			for _, w := range writers {
				if w != 1 {
					t.Fatal("duplicate output owner")
				}
			}
		}
	}
}
func TestVulkanOfflineGELUErfBindings(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	mockMemory(t)
	k.numBuffers = 2
	k.pushSize = 4
	op := &VkGELUErfF32{kernel: k}
	a := mustArena(t, 64)
	x := mustTensor(t, a, 2, 2)
	out := mustTensor(t, a, 2, 2)
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
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		if s != 0x20 || off != 0 || n != 4 {
			t.Fatal("push header")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 1)...)
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x); err != nil {
		t.Fatal(err)
	}
	h := uint64(a.state.buffer.buf)
	if !reflect.DeepEqual(ranges, [][3]uint64{{h, 0, 16}, {h, 16, 16}}) || !reflect.DeepEqual(push, []uint32{4}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	if err := op.Forward(context.Background(), x, x); err != nil {
		t.Fatal(err)
	}
	if ranges[0] != ranges[1] {
		t.Fatal("exact alias")
	}
	stage, err := op.Stage(context.Background(), out, x)
	if err != nil {
		t.Fatal(err)
	}
	stage.PushWords[0] = 99
	stage.Tensors[0] = nil
	again, err := op.Stage(context.Background(), out, x)
	if err != nil || again.PushWords[0] != 4 || again.Tensors[0] != x {
		t.Fatal("borrowed stage")
	}
	a.Close()
	op.Close()
}
func TestVulkanOfflineGELUErfAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	k.numBuffers = 2
	k.pushSize = 4
	op := &VkGELUErfF32{kernel: k}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("not a dispatch")
	})
	for _, n := range []int{1, 255, 256, 257, 513} {
		x, out := linearMetadataTensor(1, n), linearMetadataTensor(2, n)
		s, err := op.Stage(context.Background(), out, x)
		if err != nil || s.Groups[0] != uint32((n+255)/256) || s.PushWords[0] != uint32(n) {
			t.Fatal(s, err)
		}
	}
	x, out := linearMetadataTensor(1, 2, 2), linearMetadataTensor(2, 2, 2)
	for _, mutate := range []func(*VkTensorF32){func(v *VkTensorF32) { v.rank = 9 }, func(v *VkTensorF32) { v.rank = 1 }, func(v *VkTensorF32) { v.shape[0] = 0 }, func(v *VkTensorF32) { v.shape[1] = 3 }, func(v *VkTensorF32) { v.size = 12 }} {
		v := *out
		mutate(&v)
		if _, err := op.Stage(context.Background(), &v, x); err == nil {
			t.Fatal("bad output shape")
		}
	}
	aliasInput := linearMetadataTensor(7, 8)
	partial := *aliasInput
	partial.offset = 16 // Aligned but overlapping: must reject independently of alignment.
	aliasInput.arena.buffer.size = 48
	if _, err := op.Stage(context.Background(), &partial, aliasInput); err == nil {
		t.Fatal("partial alias")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, out, x)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupCount[0] = 1
	if _, err := op.Stage(context.Background(), linearMetadataTensor(3, 257), linearMetadataTensor(4, 257)); err == nil {
		t.Fatal("grid limit")
	}
	if len(lane.events) != 0 {
		t.Fatal("bad admission native calls")
	}
	if err := (*VkGELUErfF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := new(VkGELUErfF32).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := new(VkGELUErfF32).Stage(context.Background(), nil, nil); err == nil {
		t.Fatal("zero operator")
	}
	if _, err := op.Stage(context.Background(), nil, x); err == nil {
		t.Fatal("nil output")
	}
	if _, err := op.Stage(context.Background(), out, nil); err == nil {
		t.Fatal("nil input")
	}
	for rank := 1; rank <= 8; rank++ {
		shape := make([]int, rank)
		for i := range shape {
			shape[i] = 1
		}
		if _, err := op.Stage(context.Background(), linearMetadataTensor(8, shape...), linearMetadataTensor(9, shape...)); err != nil {
			t.Fatal("valid rank", rank, err)
		}
	}
	bad := *x
	bad.rank = 0
	if _, err := op.Stage(context.Background(), &bad, &bad); err == nil {
		t.Fatal("zero rank")
	}
	bad = *x
	bad.shape[0] = 0
	if _, err := op.Stage(context.Background(), &bad, &bad); err == nil {
		t.Fatal("zero dimension")
	}
	bad = *x
	bad.shape[0], bad.shape[1] = int(^uint(0)>>1), 2
	if _, err := op.Stage(context.Background(), &bad, &bad); err == nil {
		t.Fatal("shape product overflow")
	}
	// Metadata only: no allocations/dispatch at the uint32 element boundary.
	tooLarge := linearMetadataTensor(10, 65536, 65536)
	if _, err := op.Stage(context.Background(), tooLarge, tooLarge); err == nil {
		t.Fatal("uint32 element overflow")
	}
	vkLimits.WorkgroupCount[0] = ^uint32(0)
	vkLimits.StorageBufferRange = ^uint32(0)
	maxStorage := linearMetadataTensor(11, int(^uint32(0)/4))
	stage, err := op.Stage(context.Background(), maxStorage, maxStorage)
	if err != nil || stage.Groups[0] != 4194304 || stage.PushWords[0] != 1073741823 {
		t.Fatal("largest descriptor", stage, err)
	}
	oneMore := linearMetadataTensor(12, 1073741824)
	if _, err := op.Stage(context.Background(), oneMore, oneMore); err == nil {
		t.Fatal("descriptor range overflow")
	}
	if len(lane.events) != 0 {
		t.Fatal("boundary checks invoked native")
	}
}
func TestVulkanOfflineGELUErfPlan(t *testing.T) {
	m, k, a, _ := newPlanMock(t)
	a.Close()
	k.numBuffers = 4
	k.pushSize = 12
	mockMemory(t)
	a, out, x, w, b := linearMockTensors(t)
	linear := &VkLinearF32{kernel: k}
	geluKernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 2, pushSize: 4}
	gelu := &VkGELUErfF32{kernel: geluKernel}
	first, err := linear.Stage(context.Background(), out, x, w, b)
	if err != nil {
		t.Fatal(err)
	}
	second, err := gelu.Stage(context.Background(), out, out)
	if err != nil {
		t.Fatal(err)
	}
	var pushSizes []uint32
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, s, off, n uint32, p unsafe.Pointer) {
		pushSizes = append(pushSizes, n)
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
	if !reflect.DeepEqual(pushSizes, []uint32{12, 4}) {
		t.Fatal("plan push sizes")
	}
	expectErrorIs(t, gelu.Close(), ErrVulkanInFlight)
	expectErrorIs(t, linear.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	gelu.Close()
	linear.Close()
	a.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineGELUErfContract(t *testing.T) {
	offlineVK(t)
	c, err := InspectVulkanShader(spirv_gelu_erf_f32)
	if err != nil || c != (VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, StorageBindings: 3, PushBytes: 4}) {
		t.Fatal(c, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkGELUErfF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkGELUErfF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	// Keep extinst admission fail-closed. FAbs/Exp are newly used together; Tanh
	// is21 (not27), and is not in this parser's current accepted instruction set.
	for _, id := range []uint32{21, 999} {
		bad := editSPIRV(spirv_gelu_erf_f32, 12, func(a []uint32) { a[4] = id })
		_, err := InspectVulkanShader(bad)
		expectErrorIs(t, err, ErrVulkanShaderContract)
	}
	words := spirvTestWords(spirv_gelu_erf_f32)
	precise := map[uint32]bool{}
	arithmetic := []uint32{}
	comparisons := map[uint16]bool{}
	for i := 5; i < len(words); {
		n, op := int(words[i]>>16), uint16(words[i])
		if op == 71 && words[i+2] == 42 {
			precise[words[i+1]] = true
			bad := append([]uint32(nil), words[:i]...)
			bad = append(bad, uint32(4<<16)|71, words[i+1], 42, 0) // NoContraction has no operand.
			bad = append(bad, words[i+n:]...)
			_, err := InspectVulkanShader(spirvTestBytes(bad))
			expectErrorIs(t, err, ErrVulkanShaderContract)
		}
		if op == 129 || op == 131 || op == 133 || op == 136 {
			arithmetic = append(arithmetic, words[i+2])
		}
		if op == 184 || op == 188 || op == 190 {
			comparisons[op] = true
			bad := append([]uint32(nil), words[:i]...)
			bad = append(bad, uint32(4<<16)|uint32(op))
			bad = append(bad, words[i+1:i+4]...)
			bad = append(bad, words[i+n:]...)
			_, err := InspectVulkanShader(spirvTestBytes(bad))
			expectErrorIs(t, err, ErrVulkanShaderContract)
		}
		i += n
	}
	if len(arithmetic) != 20 || len(comparisons) != 3 {
		t.Fatal("shader arithmetic envelope", arithmetic, comparisons)
	}
	for _, id := range arithmetic {
		if !precise[id] {
			t.Fatal("contraction permitted for arithmetic result", id)
		}
	}
}

func TestVulkanOfflineGELUErfCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{256, 1, 1}) || c.SharedBytes != 0 || c.StorageBindings != 3 || c.PushBytes != 4 {
			t.Fatal("native shader", c, err)
		}
		*out = 1
		return VK_SUCCESS
	})
	mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
		if *(*uint32)(unsafe.Add(p, 20)) != 2 {
			t.Fatal("bindings")
		}
		*out = 2
		return VK_SUCCESS
	})
	mockVK(t, &vkCreatePipelineLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult {
		r := *(*unsafe.Pointer)(unsafe.Add(p, 40))
		if *(*uint32)(unsafe.Add(r, 8)) != 4 {
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
	op, err := NewVkGELUErfF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.kernel.numBuffers != 2 || op.kernel.pushSize != 4 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
