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

func TestVulkanNativeLinearQ8Weight(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_Q8_WEIGHT") != "1" {
		t.Skip("explicit Q8-weight Vulkan qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("Q8-weight Vulkan qualification requires timeout<=2m")
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
		x, weight, bias := nativeData(m*k, 111, .5), nativeData(n*k, 112, .05), nativeData(n, 113, .1)
		packed, scales, err := packLinearQ8Weights(weight, n, k)
		if err != nil {
			t.Fatal(err)
		}
		dequant := unpackLinearQ8Weights(packed, scales, n, k)
		op, err := NewVkLinearQ8WeightF32(context.Background(), weight, n, k)
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
		dequantRef := nativeLinearRef(x, dequant, bias, m, k, n)
		nativeCompare(t, fmt.Sprintf("linear-q8-weight%v", dims), got, dequantRef, 2e-5, 2e-5, elapsed)
		originalRef := nativeLinearRef(x, weight, bias, m, k, n)
		maxQuant, rmsQuant, rmsRef := 0.0, 0.0, 0.0
		for i, v := range got {
			d := float64(v) - originalRef[i]
			maxQuant = math.Max(maxQuant, math.Abs(d))
			rmsQuant += d * d
			rmsRef += originalRef[i] * originalRef[i]
		}
		rmsQuant = math.Sqrt(rmsQuant / float64(len(got)))
		rmsRef = math.Sqrt(rmsRef / float64(len(got)))
		metric, _ := json.Marshal(map[string]any{"shape": dims, "values": len(got), "max_original_f32_abs": maxQuant, "rms_error": rmsQuant, "reference_rms": rmsRef, "rms_ratio": rmsQuant / max(rmsRef, math.SmallestNonzeroFloat64)})
		t.Log("Q8_WEIGHT_QUALITY " + string(metric))
		left()
		right()
		if op.storage == nil || op.storage.size >= uint64(len(weight)*4) && len(weight) > 2 {
			t.Fatal("Q8 storage extent", dims, op.storage)
		}
		op.Close()
		a.Close()
	}
	after := VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain || after.DeviceLost {
		t.Fatal("Q8 native leak", before, after)
	}
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_Q8_TIMING") != "1" {
		return
	}
	for _, dims := range [][3]int{{1500, 1280, 1280}, {1500, 1280, 5120}, {1500, 5120, 1280}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, weight, bias := nativeData(m*k, 121, .125), nativeData(n*k, 122, .05), nativeData(n, 123, .125)
		packed, scales, err := packLinearQ8Weights(weight, n, k)
		if err != nil {
			t.Fatal(err)
		}
		dequant := unpackLinearQ8Weights(packed, scales, n, k)
		candidate, err := NewVkLinearQ8WeightF32(context.Background(), weight, n, k)
		if err != nil {
			t.Fatal(err)
		}
		baseline, err := NewVkLinearF32(context.Background())
		if err != nil {
			candidate.Close()
			t.Fatal(err)
		}
		regtile, err := NewVkLinearRegTileF32(context.Background())
		if err != nil {
			candidate.Close()
			baseline.Close()
			t.Fatal(err)
		}
		a, err := NewVkTensorArena(context.Background(), 96<<20)
		if err != nil {
			candidate.Close()
			baseline.Close()
			regtile.Close()
			t.Fatal(err)
		}
		tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, dequant, n, k), nativeTensor(t, a, bias, n)
		out := nativeTensor(t, a, nil, m, n)
		run := func(name string) int64 {
			start := time.Now()
			var err error
			switch name {
			case "q8_weight":
				err = candidate.Forward(context.Background(), out, tx, tb)
			case "f32_regtile":
				err = regtile.Forward(context.Background(), out, tx, tw, tb)
			default:
				err = baseline.Forward(context.Background(), out, tx, tw, tb)
			}
			if err != nil {
				t.Fatal(err)
			}
			return time.Since(start).Nanoseconds()
		}
		run("f32")
		reference := nativeDownload(t, out)
		maxAbs := map[string]float64{}
		for _, name := range []string{"f32_regtile", "q8_weight"} {
			run(name)
			got := nativeDownload(t, out)
			for i, v := range got {
				d := math.Abs(float64(v) - float64(reference[i]))
				maxAbs[name] = math.Max(maxAbs[name], d)
				if d > 2e-5+2e-5*math.Abs(float64(reference[i])) {
					t.Fatal(name+"/F32 dequant-weight mismatch", dims, i, v, reference[i])
				}
			}
		}
		samples := map[string][]int64{"f32": {}, "f32_regtile": {}, "q8_weight": {}}
		orders := [][]string{
			{"f32", "f32_regtile", "q8_weight", "q8_weight", "f32_regtile", "f32"},
			{"q8_weight", "f32_regtile", "f32", "f32", "f32_regtile", "q8_weight"},
			{"f32_regtile", "q8_weight", "f32", "f32", "q8_weight", "f32_regtile"},
			{"f32", "q8_weight", "f32_regtile", "f32_regtile", "q8_weight", "f32"},
		}
		for _, order := range orders {
			for _, name := range order {
				samples[name] = append(samples[name], run(name))
			}
		}
		medians := map[string]int64{}
		for name, values := range samples {
			sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
			medians[name] = (values[3] + values[4]) / 2
		}
		record, _ := json.Marshal(map[string]any{
			"device": device, "shape": dims, "f32_weight_bytes": len(weight) * 4,
			"q8_storage_bytes": candidate.storage.size, "median_ns": medians,
			"q8_vs_f32":                    float64(medians["f32"]) / float64(medians["q8_weight"]),
			"q8_vs_f32_regtile":            float64(medians["f32_regtile"]) / float64(medians["q8_weight"]),
			"f32_regtile_vs_f32":           float64(medians["f32"]) / float64(medians["f32_regtile"]),
			"max_dequant_f32_baseline_abs": maxAbs, "gpu_timestamps": false,
		})
		t.Log("Q8_WEIGHT_TIMING " + string(record))
		candidate.Close()
		baseline.Close()
		regtile.Close()
		a.Close()
		final := VulkanMemoryStats()
		if final.Bytes != before.Bytes || final.Allocations != before.Allocations {
			t.Fatal("timed Q8 leak", dims, before, final)
		}
	}
}
