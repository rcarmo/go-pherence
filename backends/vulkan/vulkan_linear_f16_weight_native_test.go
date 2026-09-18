package vulkan

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativeLinearF16Weight(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_F16_WEIGHT") != "1" {
		t.Skip("explicit packed-F16 Vulkan qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("packed-F16 Vulkan qualification requires timeout<=2m")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	lower := strings.ToLower(device)
	if want == "" || !strings.Contains(device, want) || strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("unexpected physical device", device)
	}
	before := VulkanMemoryStats()
	for _, dims := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {17, 31, 19}, {33, 65, 63}, {3, 384, 9}, {2, 1280, 7}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, weight, bias := nativeData(m*k, 81, .5), nativeData(n*k, 82, .5), nativeData(n, 83, .1)
		packed, err := packLinearF16Weights(weight)
		if err != nil {
			t.Fatal(err)
		}
		rounded := unpackLinearF16Weights(packed, len(weight))
		op, err := NewVkLinearF16WeightF32(context.Background(), weight, n, k)
		if err != nil {
			t.Fatal(err)
		}
		a, err := NewVkTensorArena(context.Background(), (m*k+m*n+n)*4+4096)
		if err != nil {
			op.Close()
			t.Fatal(err)
		}
		left := nativeGuard(t, a)
		tx, tb, out := nativeTensor(t, a, x, m, k), nativeTensor(t, a, bias, n), nativeTensor(t, a, nil, m, n)
		right := nativeGuard(t, a)
		elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tx, tb) })
		got := nativeDownload(t, out)
		nativeCompare(t, fmt.Sprintf("linear-f16-weight%v", dims), got, nativeLinearRef(x, rounded, bias, m, k, n), 2e-5, 2e-5, elapsed)
		left()
		right()
		if op.weight == nil || op.weight.size != uint64(len(packed)*4) || op.weight.size >= uint64(len(weight)*4) && len(weight) > 1 {
			t.Fatal("packed weight extent", dims, op.weight)
		}
		if err := op.Close(); err != nil {
			t.Fatal(err)
		}
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
	}
	after := VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain || after.DeviceLost {
		t.Fatal("packed-F16 native leak", before, after)
	}
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_F16_TIMING") != "1" {
		return
	}
	// One encoder projection shape. Same rounded weights and tensors for both
	// kernels; timing excludes construction, packing, upload and download.
	m, k, n := 1500, 1280, 1280
	x, weight, bias := nativeData(m*k, 91, .125), nativeData(n*k, 92, .125), nativeData(n, 93, .125)
	packed, err := packLinearF16Weights(weight)
	if err != nil {
		t.Fatal(err)
	}
	rounded := unpackLinearF16Weights(packed, len(weight))
	candidate, err := NewVkLinearF16WeightF32(context.Background(), weight, n, k)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := NewVkLinearF32(context.Background())
	if err != nil {
		candidate.Close()
		t.Fatal(err)
	}
	a, err := NewVkTensorArena(context.Background(), 32<<20)
	if err != nil {
		candidate.Close()
		baseline.Close()
		t.Fatal(err)
	}
	tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, rounded, n, k), nativeTensor(t, a, bias, n)
	out := nativeTensor(t, a, nil, m, n)
	run := func(name string) int64 {
		start := time.Now()
		var err error
		if name == "packed_f16" {
			err = candidate.Forward(context.Background(), out, tx, tb)
		} else {
			err = baseline.Forward(context.Background(), out, tx, tw, tb)
		}
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(start).Nanoseconds()
	}
	run("f32")
	reference := nativeDownload(t, out)
	run("packed_f16")
	got := nativeDownload(t, out)
	maxAbs := 0.0
	for i, v := range got {
		d := math.Abs(float64(v) - float64(reference[i]))
		maxAbs = math.Max(maxAbs, d)
		if d > 2e-5+2e-5*math.Abs(float64(reference[i])) {
			t.Fatal("packed/F32 rounded-weight mismatch", i, v, reference[i])
		}
	}
	samples := map[string][]int64{"f32": {}, "packed_f16": {}}
	for block := 0; block < 3; block++ {
		order := []string{"f32", "packed_f16", "packed_f16", "f32"}
		if block&1 != 0 {
			order = []string{"packed_f16", "f32", "f32", "packed_f16"}
		}
		for _, name := range order {
			samples[name] = append(samples[name], run(name))
		}
	}
	medians := map[string]int64{}
	for name, values := range samples {
		sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
		medians[name] = (values[2] + values[3]) / 2
	}
	record, _ := json.Marshal(map[string]any{"device": device, "shape": []int{m, k, n}, "f32_weight_bytes": len(weight) * 4, "packed_weight_bytes": len(packed) * 4, "median_ns": medians, "speedup": float64(medians["f32"]) / float64(medians["packed_f16"]), "max_f32_baseline_abs": maxAbs, "gpu_timestamps": false})
	t.Log("F16_WEIGHT_TIMING " + string(record))
	candidate.Close()
	baseline.Close()
	a.Close()
	final := VulkanMemoryStats()
	if final.Bytes != before.Bytes || final.Allocations != before.Allocations {
		t.Fatal("timed packed-F16 leak", before, final)
	}
}
