package vulkan

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
	"unsafe"
)

func linearRegModel(t *testing.T, x, w, b []float32, m, k, n int, seed int64) []float32 {
	t.Helper()
	gx, gy := (n+31)/32, (m+31)/32
	out := make([]float32, m*n)
	writers := make([]int, len(out))
	rng := rand.New(rand.NewSource(seed))
	lanes := rng.Perm(256)
	for _, g := range rng.Perm(gx * gy) {
		rowBase, colBase := (g/gx)*32, (g%gx)*32
		var sums [256][4]float32
		for base := 0; base < k; base += 32 {
			var xt, wt [1024]float32
			var owners [1024]int
			for _, lane := range lanes {
				for i := lane; i < 1024; i += 256 {
					r, kk := i/32, base+i%32
					owners[i]++
					if kk < k {
						if rowBase+r < m {
							xt[i] = x[(rowBase+r)*k+kk]
						}
						if colBase+r < n {
							wt[i] = w[(colBase+r)*k+kk]
						}
					}
				}
			}
			for _, o := range owners {
				if o != 1 {
					t.Fatal("sharedtile owner")
				}
			}
			for _, lane := range lanes {
				cx, ry := lane%16, lane/16
				for j := 0; j < 32; j++ {
					a0, a1 := xt[ry*32+j], xt[(ry+16)*32+j]
					b0, b1 := wt[cx*32+j], wt[(cx+16)*32+j]
					sums[lane][0] += a0 * b0
					sums[lane][1] += a0 * b1
					sums[lane][2] += a1 * b0
					sums[lane][3] += a1 * b1
				}
			}
		}
		for _, lane := range lanes {
			cx, ry := lane%16, lane/16
			for v := 0; v < 4; v++ {
				r, c := rowBase+ry+(v/2)*16, colBase+cx+(v%2)*16
				if r < m && c < n {
					out[r*n+c] = sums[lane][v] + b[c]
					writers[r*n+c]++
				}
			}
		}
	}
	for _, v := range writers {
		if v != 1 {
			t.Fatal("outputowner", v)
		}
	}
	return out
}
func TestVulkanOfflineLinearRegTile(t *testing.T) {
	for _, dims := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {16, 17, 15}, {17, 31, 19}, {31, 33, 7}, {33, 17, 31}, {3, 384, 9}, {2, 1280, 7}, {1, 16384, 2}, {32, 32, 32}, {33, 65, 63}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, w, b := nativeData(m*k, 11, .5), nativeData(n*k, 12, .5), nativeData(n, 13, .1)
		ref := nativeLinearRef(x, w, b, m, k, n)
		for seed := int64(0); seed < 4; seed++ {
			got := linearRegModel(t, x, w, b, m, k, n, seed)
			old, _ := linearTileModel(x, w, b, m, k, n, seed)
			for i, v := range got {
				if math.Float32bits(v) != math.Float32bits(old[i]) || math.Abs(float64(v)-ref[i]) > 2e-5+2e-5*math.Abs(ref[i]) {
					t.Fatal(dims, seed, i, v, old[i], ref[i])
				}
			}
		}
	}
}
func TestVulkanOfflineLinearRegAdmission(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	k.numBuffers = 4
	k.pushSize = 12
	op := &VkLinearF32{kernel: k, outputTile: 32}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("not dispatch")
	})
	for _, d := range [][3]int{{1, 1, 1}, {31, 17, 33}, {32, 32, 32}, {33, 31, 63}, {16384, 1, 16384}} {
		m, in, n := d[0], d[1], d[2]
		vkLimits.StorageBufferRange = math.MaxUint32
		s, err := op.Stage(context.Background(), linearMetadataTensor(1, m, n), linearMetadataTensor(2, m, in), linearMetadataTensor(3, n, in), linearMetadataTensor(4, n))
		if err != nil || s.Groups != ([3]uint32{uint32((n + 31) / 32), uint32((m + 31) / 32), 1}) {
			t.Fatal(d, s, err)
		}
	}
	op.outputTile = 31
	if _, err := op.Stage(context.Background(), linearMetadataTensor(1, 1, 1), linearMetadataTensor(2, 1, 1), linearMetadataTensor(3, 1, 1), linearMetadataTensor(4, 1)); err == nil {
		t.Fatal("invalidtile")
	}
	if len(lane.events) != 0 {
		t.Fatal("mutation")
	}
	c, err := InspectVulkanShader(spirv_linear_f32_regtile)
	if err != nil || c != (VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 8192, StorageBindings: 15, PushBytes: 12}) {
		t.Fatal(c, err)
	}
	vkLimits = offlineLimits()
	vkLimits.SharedMemoryBytes = 8191
	_, err = NewVkLinearRegTileF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkLinearRegTileF32(ctx)
	expectErrorIs(t, err, context.Canceled)
}
func TestVulkanNativeLinearRegTile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_REGTILE") != "1" {
		t.Skip("explicit candidate GPU window required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("timeout<=120s required")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(VulkanDeviceName(), want) {
		t.Fatal("unexpectedGPU")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	old, err := NewVkLinearF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, old)
	candidate, err := NewVkLinearRegTileF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, candidate)
	if !t.Run("numerics", func(t *testing.T) {
		a := nativeArena(t)
		for _, d := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {16, 17, 15}, {17, 31, 19}, {31, 33, 7}, {33, 17, 31}, {3, 384, 9}, {2, 1280, 7}, {1, 16384, 2}, {32, 32, 32}, {33, 65, 63}} {
			m, k, n := d[0], d[1], d[2]
			x, w, b := nativeData(m*k, 11, .5), nativeData(n*k, 12, .5), nativeData(n, 13, .1)
			left := nativeGuard(t, a)
			tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, w, n, k), nativeTensor(t, a, b, n)
			out, ref := nativeTensor(t, a, nil, m, n), nativeTensor(t, a, nil, m, n)
			right := nativeGuard(t, a)
			elapsed := nativeRun(t, func(ctx context.Context) error { return candidate.Forward(ctx, out, tx, tw, tb) })
			got := nativeDownload(t, out)
			nativeCompare(t, fmt.Sprint("regtile", d), got, nativeLinearRef(x, w, b, m, k, n), 2e-5, 2e-5, elapsed)
			nativeRun(t, func(ctx context.Context) error { return old.Forward(ctx, ref, tx, tw, tb) })
			oldValues := nativeDownload(t, ref)
			max := 0.
			for i, v := range got {
				e := math.Abs(float64(v) - float64(oldValues[i]))
				max = math.Max(max, e)
				if e > 2e-5+2e-5*math.Abs(float64(oldValues[i])) {
					t.Fatal("candidate baseline mismatch", d, i, v, oldValues[i])
				}
			}
			t.Logf("REGTILE_NUMERIC shape=%v max_baseline_abs=%g", d, max)
			left()
			right()
		}
	}) {
		return
	}
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_TIMING") != "1" {
		return
	}
	for _, d := range [][3]int{{1500, 1280, 1280}, {1500, 1280, 5120}, {1500, 5120, 1280}} {
		if !t.Run(fmt.Sprint(d), func(t *testing.T) {
			m, k, n := d[0], d[1], d[2]
			x, w, b := nativeData(m*k, 11, .125), nativeData(n*k, 12, .125), nativeData(n, 13, .125)
			a, err := NewVkTensorArena(context.Background(), 96<<20)
			if err != nil {
				t.Fatal(err)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, w, n, k), nativeTensor(t, a, b, n)
			out := nativeTensor(t, a, nil, m, n)
			right := nativeGuard(t, a)
			run := func(op *VkLinearF32) time.Duration {
				return nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tx, tw, tb) })
			}
			run(old)
			reference := nativeDownload(t, out)
			run(candidate)
			values := nativeDownload(t, out)
			max := 0.
			for i, v := range values {
				e := math.Abs(float64(v) - float64(reference[i]))
				max = math.Max(max, e)
				if math.IsNaN(float64(v)) || e > 2e-5+2e-5*math.Abs(float64(reference[i])) {
					t.Fatal("fullshape mismatch", i, v, reference[i])
				}
			}
			left()
			right()
			// Identical input tensors, 4 alternated ABBA blocks, 8 timings per kernel.
			times := map[string][]int64{"baseline": {}, "candidate": {}}
			for block := 0; block < 4; block++ {
				order := []string{"baseline", "candidate", "candidate", "baseline"}
				if block%2 == 1 {
					order = []string{"candidate", "baseline", "baseline", "candidate"}
				}
				for _, name := range order {
					op := old
					if name == "candidate" {
						op = candidate
					}
					ns := run(op).Nanoseconds()
					// Transfers/checks are outside timing; every sample must still
					// produce identical values, not just the initial warm-up.
					got := nativeDownload(t, out)
					for i, v := range got {
						if math.Float32bits(v) != math.Float32bits(reference[i]) {
							t.Fatal("timed output mismatch", name, block, i)
						}
					}
					left()
					right()
					times[name] = append(times[name], ns)
					record, _ := json.Marshal(map[string]any{"shape": d, "block": block, "kernel": name, "wall_ns": ns})
					t.Log("REGTILE_SAMPLE " + string(record))
				}
			}
			medians := map[string]float64{}
			for name, ns := range times {
				sort.Slice(ns, func(i, j int) bool { return ns[i] < ns[j] })
				medians[name] = float64(ns[3]+ns[4]) / 2
			}
			record, _ := json.Marshal(map[string]any{"shape": d, "max_baseline_abs": max, "median_ns": medians, "speedup": medians["baseline"] / medians["candidate"], "gpu_timestamps": false})
			t.Log("REGTILE_TIMING " + string(record))
			left()
			right()
		}) {
			return
		}
	}
}
func nativeEncoderMemoryCheck(t *testing.T, before VulkanMemoryUsage) {
	t.Helper()
	after := VulkanMemoryStats()
	if before.Bytes != after.Bytes || before.Allocations != after.Allocations {
		t.Fatal("nativecandidate leak", before, after)
	}
}

func TestVulkanOfflineLinearRegCreationSuccess(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
		n := *(*uint64)(unsafe.Add(p, 24))
		ptr := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		c, err := vkInspectSPIRV(unsafe.Slice((*uint32)(ptr), int(n)/4))
		if err != nil || c.LocalSize != ([3]uint32{16, 16, 1}) || c.SharedBytes != 8192 || c.StorageBindings != 15 || c.PushBytes != 12 {
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
	op, err := NewVkLinearRegTileF32(context.Background())
	if err != nil || op == nil {
		t.Fatal(err)
	}
	if op.outputTile != 32 || op.kernel.numBuffers != 4 || op.kernel.pushSize != 12 || freedShader != 1 {
		t.Fatal("construction")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
