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

// Opt-in real-device qualification; never included in the mock selection.
// Run in its own process, not with legacy Vulkan tests that retain resources.
// A failure stops later subtests (no automatic device retry/reinitialisation).
func TestVulkanNativeSpeech(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_SPEECH") != "1" {
		t.Skip("set GO_PHERENCE_TEST_VULKAN_SPEECH=1 in an authorised compute window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 3*time.Minute {
		t.Fatal("native qualification requires go test -timeout of at most 3m")
	}
	if !VulkanInit() {
		t.Fatal("native Vulkan initialisation failed")
	}
	device := VulkanDeviceName()
	// Check physical properties, not just a renderer-name heuristic. Types1/2
	// are integrated/discrete GPU; CPU/virtual/other devices are not this gate.
	var props vkDeviceProperties
	func() {
		if err := vkAcquire(context.Background()); err != nil {
			t.Fatal(err)
		}
		defer vkRelease()
		vkGetPhysicalDeviceProperties(vkPhysDev, unsafe.Pointer(&props))
	}()
	if props.deviceType != 1 && props.deviceType != 2 {
		t.Fatal("not an integrated/discrete GPU", props.deviceType, device)
	}
	if strings.Contains(strings.ToLower(device), "llvmpipe") || strings.Contains(strings.ToLower(device), "lavapipe") {
		t.Fatal("software Vulkan is not hardware qualification", device)
	}
	if want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE"); want != "" && !strings.Contains(device, want) {
		t.Fatal("unexpected Vulkan device", device, want)
	}
	limits, err := VulkanLimits()
	if err != nil {
		t.Fatal(err)
	}
	meta, _ := json.Marshal(map[string]any{"device": device, "limits": limits, "kind": "native_device", "vendor_id": props.vendorID, "device_id": props.deviceID, "device_type": props.deviceType, "driver_version": props.driverVersion})
	t.Log(string(meta))
	before := VulkanMemoryStats()
	for _, tc := range []struct {
		name string
		run  func(*testing.T)
	}{
		{"GELU", nativeSpeechGELU}, {"Linear", nativeSpeechLinear}, {"LayerNorm", nativeSpeechLayerNorm}, {"Attention", nativeSpeechAttention}, {"Plan", nativeSpeechPlan},
	} {
		if !t.Run(tc.name, tc.run) {
			return
		}
		if !VulkanReady() {
			t.Fatal("device not ready after", tc.name)
		}
		after := VulkanMemoryStats()
		if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
			t.Fatalf("allocation leak after %s: before%+v after%+v", tc.name, before, after)
		}
	}
}

type nativeSpeechMetric struct {
	Name      string  `json:"name"`
	Values    int     `json:"values"`
	MaxAbs    float64 `json:"max_abs"`
	AbsBudget float64 `json:"abs_budget"`
	RelBudget float64 `json:"rel_budget"`
	WallNS    int64   `json:"wall_ns"`
}

func nativeCompare(t *testing.T, name string, got []float32, want []float64, abs, rel float64, elapsed time.Duration) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("reference length")
	}
	metric := nativeSpeechMetric{Name: name, Values: len(got), AbsBudget: abs, RelBudget: rel, WallNS: int64(elapsed)}
	for i, v := range got {
		e := math.Abs(float64(v) - want[i])
		metric.MaxAbs = math.Max(metric.MaxAbs, e)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || e > abs+rel*math.Abs(want[i]) {
			t.Fatalf("%s index%d got%.10g want%.10g error%.5g budget%.5g", name, i, v, want[i], e, abs+rel*math.Abs(want[i]))
		}
	}
	b, _ := json.Marshal(metric)
	t.Log("NATIVE_METRIC " + string(b))
}
func nativeClose(t *testing.T, c interface{ Close() error }) {
	t.Helper()
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Errorf("native cleanup: %v", err)
		}
	})
}
func nativeArena(t *testing.T) *VkTensorArena {
	t.Helper()
	a, err := NewVkTensorArena(context.Background(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, a)
	return a
}
func nativeTensor(t *testing.T, a *VkTensorArena, data []float32, shape ...int) *VkTensorF32 {
	t.Helper()
	v, err := a.AllocF32(context.Background(), shape...)
	if err != nil {
		t.Fatal(err)
	}
	if data != nil {
		if err := v.Upload(context.Background(), data); err != nil {
			t.Fatal(err)
		}
	}
	return v
}
func nativeDownload(t *testing.T, v *VkTensorF32) []float32 {
	t.Helper()
	out := make([]float32, v.Elements())
	if err := v.Download(context.Background(), out); err != nil {
		t.Fatal(err)
	}
	return out
}
func nativeGuard(t *testing.T, a *VkTensorArena) func() {
	t.Helper()
	data := make([]float32, 256)
	for i := range data {
		data[i] = float32(i) + 7654
	}
	v := nativeTensor(t, a, data, len(data))
	return func() {
		t.Helper()
		out := nativeDownload(t, v)
		for i := range out {
			if out[i] != data[i] {
				t.Fatal("native arena guard overwritten", i)
			}
		}
	}
}
func nativeRun(t *testing.T, fn func(context.Context) error) time.Duration {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := time.Now()
	if err := fn(ctx); err != nil {
		t.Fatal("native dispatch", err)
	}
	return time.Since(start)
}
func nativeData(n int, seed int64, scale float32) []float32 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(r.Float64()*2-1) * scale
	}
	return a
}
func nativeF32(a []float64) []float32 {
	out := make([]float32, len(a))
	for i, v := range a {
		out[i] = float32(v)
	}
	return out
}
func nativeLinearRef(x, w, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			s := float64(b[c])
			for d := 0; d < k; d++ {
				s += float64(x[r*k+d]) * float64(w[c*k+d])
			}
			out[r*n+c] = s
		}
	}
	return out
}
func nativeLNRef(x, w, b []float32, rows, width int, eps float32) []float64 {
	out := make([]float64, len(x))
	for r := 0; r < rows; r++ {
		mean := 0.
		for _, v := range x[r*width : (r+1)*width] {
			mean += float64(v)
		}
		mean /= float64(width)
		variance := 0.
		for _, v := range x[r*width : (r+1)*width] {
			d := float64(v) - mean
			variance += d * d
		}
		inv := 1 / math.Sqrt(variance/float64(width)+float64(eps))
		for c := 0; c < width; c++ {
			out[r*width+c] = (float64(x[r*width+c])-mean)*inv*float64(w[c]) + float64(b[c])
		}
	}
	return out
}
func nativeGELURef(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = .5 * float64(v) * math.Erfc(-float64(v)/math.Sqrt2)
	}
	return out
}

func nativeSpeechGELU(t *testing.T) {
	op, err := NewVkGELUErfF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	values := nativeData(4097, 13, 10)
	values = append(values, 0, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, -math.MaxFloat32, 10, -10, math.Nextafter32(10, 0), math.Nextafter32(-10, 0))
	a := nativeArena(t)
	left := nativeGuard(t, a)
	x := nativeTensor(t, a, values, len(values))
	out := nativeTensor(t, a, nil, len(values))
	right := nativeGuard(t, a)
	for _, inplace := range []bool{false, true} {
		dest := out
		if inplace {
			dest = x
		}
		elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, dest, x) })
		nativeCompare(t, fmt.Sprintf("gelu/inplace=%t", inplace), nativeDownload(t, dest), nativeGELURef(values), 2e-6, 2e-6, elapsed)
		left()
		right()
	}
}
func nativeSpeechLinear(t *testing.T) {
	op, err := NewVkLinearF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	a := nativeArena(t)
	for _, s := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {16, 17, 15}, {17, 31, 19}, {31, 33, 7}, {33, 17, 31}, {3, 384, 9}, {2, 1280, 7}, {1, 16384, 2}} {
		m, k, n := s[0], s[1], s[2]
		x, w, b := nativeData(m*k, 11, .5), nativeData(n*k, 12, .5), nativeData(n, 13, .1)
		left := nativeGuard(t, a)
		tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, w, n, k), nativeTensor(t, a, b, n)
		out := nativeTensor(t, a, nil, m, n)
		right := nativeGuard(t, a)
		elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tx, tw, tb) })
		nativeCompare(t, fmt.Sprint("linear", s), nativeDownload(t, out), nativeLinearRef(x, w, b, m, k, n), 2e-5, 2e-5, elapsed)
		left()
		right()
	}
}
func nativeSpeechLayerNorm(t *testing.T) {
	op, err := NewVkLayerNormF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	a := nativeArena(t)
	for _, width := range []int{1, 3, 17, 255, 256, 257, 384, 768, 1280, 4097, 16384} {
		rows := 3
		x, w, b := nativeData(rows*width, 11, 2), nativeData(width, 12, 1), nativeData(width, 13, .1)
		left := nativeGuard(t, a)
		tx, tw, tb := nativeTensor(t, a, x, rows, width), nativeTensor(t, a, w, width), nativeTensor(t, a, b, width)
		out := nativeTensor(t, a, nil, rows, width)
		right := nativeGuard(t, a)
		for _, inplace := range []bool{false, true} {
			dest := out
			if inplace {
				dest = tx
			}
			elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, dest, tx, tw, tb, 1e-5) })
			nativeCompare(t, fmt.Sprintf("layernorm/width=%d/inplace=%t", width, inplace), nativeDownload(t, dest), nativeLNRef(x, w, b, rows, width, 1e-5), 2e-5, 2e-5, elapsed)
			left()
			right()
		}
	}
}
func nativeSpeechAttention(t *testing.T) {
	op, err := NewVkAttentionF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	a := nativeArena(t)
	for _, s := range [][4]int{{1, 1, 1, 1}, {2, 3, 2, 3}, {15, 17, 3, 7}, {16, 16, 2, 16}, {17, 31, 2, 17}, {31, 33, 3, 31}, {33, 17, 2, 32}, {3, 65, 2, 63}, {2, 129, 3, 64}, {1, 4096, 1, 1}, {4096, 1, 1, 1}, {1, 17, 32, 64}} {
		m, n, h, d := s[0], s[1], s[2], s[3]
		q, k, v := nativeData(m*h*d, 11, 2), nativeData(n*h*d, 12, 2), nativeData(n*h*d, 13, 2)
		left := nativeGuard(t, a)
		tq, tk, tv := nativeTensor(t, a, q, m, h*d), nativeTensor(t, a, k, n, h*d), nativeTensor(t, a, v, n, h*d)
		out := nativeTensor(t, a, nil, m, h*d)
		right := nativeGuard(t, a)
		elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tq, tk, tv, h) })
		nativeCompare(t, fmt.Sprint("attention", s), nativeDownload(t, out), attentionReference(q, k, v, m, n, h, d), 2e-5, 2e-5, elapsed)
		left()
		right()
	}
	for _, sign := range []float32{-1, 0, 1} {
		const m, n, h, d = 17, 49, 2, 3
		const width = h * d
		q, k, v := make([]float32, m*width), make([]float32, n*width), make([]float32, n*width)
		for i := range q {
			q[i] = sign
		}
		for j := 0; j < n; j++ {
			for c := 0; c < width; c++ {
				k[j*width+c] = []float32{-800, 900, 900, -1000}[j/16] + float32(j%3)
				v[j*width+c] = float32(j%7-3) * float32(c+1)
			}
		}
		left := nativeGuard(t, a)
		tq, tk, tv := nativeTensor(t, a, q, m, width), nativeTensor(t, a, k, n, width), nativeTensor(t, a, v, n, width)
		out := nativeTensor(t, a, nil, m, width)
		right := nativeGuard(t, a)
		elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tq, tk, tv, h) })
		nativeCompare(t, fmt.Sprint("attention-stress/", sign), nativeDownload(t, out), attentionReference(q, k, v, m, n, h, d), 2e-4, 2e-5, elapsed)
		left()
		right()
	}
}
func nativeSpeechPlan(t *testing.T) {
	if !t.Run("tiny", func(t *testing.T) { nativeSpeechPlanShape(t, 17, 6, 2, false) }) {
		return
	}
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TIMING") == "1" {
		t.Run("synthetic256x384", func(t *testing.T) { nativeSpeechPlanShape(t, 256, 384, 6, true) })
	}
}
func nativeSpeechPlanShape(t *testing.T, rows, width, heads int, measure bool) {
	linear, err := NewVkLinearF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, linear)
	ln, err := NewVkLayerNormF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, ln)
	gelu, err := NewVkGELUErfF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, gelu)
	attention, err := NewVkAttentionF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, attention)
	add, err := VkKernelCreate(spirv_vec_add_f32, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, add)
	a := nativeArena(t)
	left := nativeGuard(t, a)
	x, w, b, gamma, beta := nativeData(rows*width, 11, 1), nativeData(width*width, 12, .3), nativeData(width, 13, .1), nativeData(width, 14, .5), nativeData(width, 15, .1)
	tx, tw, tb, tg, tbeta := nativeTensor(t, a, x, rows, width), nativeTensor(t, a, w, width, width), nativeTensor(t, a, b, width), nativeTensor(t, a, gamma, width), nativeTensor(t, a, beta, width)
	z, att, out := nativeTensor(t, a, nil, rows, width), nativeTensor(t, a, nil, rows, width), nativeTensor(t, a, nil, rows, width)
	right := nativeGuard(t, a)
	var stages []VkF32Stage
	appendStage := func(s VkF32Stage, e error) {
		if e != nil {
			t.Fatal(e)
		}
		stages = append(stages, s)
	}
	appendStage(linear.Stage(context.Background(), z, tx, tw, tb))
	appendStage(ln.Stage(context.Background(), z, z, tg, tbeta, 1e-5))
	appendStage(gelu.Stage(context.Background(), z, z))
	appendStage(attention.Stage(context.Background(), att, z, z, z, heads))
	appendStage(linear.Stage(context.Background(), out, att, tw, tb))
	stages = append(stages, VkF32Stage{Kernel: add, Groups: [3]uint32{uint32((rows*width + 255) / 256), 1, 1}, Tensors: []*VkTensorF32{out, tx, out}, PushWords: []uint32{uint32(rows * width)}})
	plan, err := NewVkF32Plan(context.Background(), stages)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, plan)
	// Independent CPU pipeline, rounding each materialised intermediate to F32.
	ref := nativeF32(nativeLinearRef(x, w, b, rows, width, width))
	ref = nativeF32(nativeLNRef(ref, gamma, beta, rows, width, 1e-5))
	ref = nativeF32(nativeGELURef(ref))
	ref = nativeF32(attentionReference(ref, ref, ref, rows, rows, heads, width/heads))
	expected := nativeLinearRef(ref, w, b, rows, width, width)
	for i := range expected {
		expected[i] = float64(float32(expected[i]) + x[i])
	}
	for i := 0; i < 3; i++ {
		elapsed := nativeRun(t, plan.Run)
		nativeCompare(t, fmt.Sprint("six-stage-plan/", i), nativeDownload(t, out), expected, 5e-5, 5e-5, elapsed)
		left()
		right()
	}
	// Separate dispatch reference uses the same operators but host fences between
	// stages, testing inter-stage visibility of the one-submit plan.
	push := []uint32{uint32(rows * width)}
	separateRun := func(ctx context.Context) error {
		if err := linear.Forward(ctx, z, tx, tw, tb); err != nil {
			return err
		}
		if err := ln.Forward(ctx, z, z, tg, tbeta, 1e-5); err != nil {
			return err
		}
		if err := gelu.Forward(ctx, z, z); err != nil {
			return err
		}
		if err := attention.Forward(ctx, att, z, z, z, heads); err != nil {
			return err
		}
		if err := linear.Forward(ctx, out, att, tw, tb); err != nil {
			return err
		}
		return add.DispatchF32Context(ctx, uint32((rows*width+255)/256), 1, 1, []*VkTensorF32{out, tx, out}, unsafePushWords(push))
	}
	nativeRun(t, separateRun)
	separate := nativeDownload(t, out)
	nativeRun(t, plan.Run)
	together := nativeDownload(t, out)
	for i := range together {
		if math.Float32bits(together[i]) != math.Float32bits(separate[i]) {
			t.Fatal("plan/separate mismatch", i, together[i], separate[i])
		}
	}
	t.Log("plan vs six separate fenced dispatches: bit-exact")
	left()
	right()
	if measure {
		// Warm both equal-work paths; timestamps/creation/uploads/downloads excluded.
		// ABBA reverses first/last order in alternate blocks. These are host-wall
		// submission+wait timings, not device timestamps or trained-model throughput.
		for i := 0; i < 2; i++ {
			nativeRun(t, plan.Run)
			nativeRun(t, separateRun)
		}
		type sample struct {
			Block  int    `json:"block"`
			Path   string `json:"path"`
			WallNS int64  `json:"wall_ns"`
		}
		samples := []sample{}
		byPath := map[string][]int64{"plan": {}, "separate": {}}
		for block := 0; block < 6; block++ {
			order := []string{"plan", "separate", "separate", "plan"}
			if block%2 == 1 {
				order = []string{"separate", "plan", "plan", "separate"}
			}
			for _, path := range order {
				fn := plan.Run
				if path == "separate" {
					fn = separateRun
				}
				duration := nativeRun(t, fn)
				samples = append(samples, sample{block, path, int64(duration)})
				byPath[path] = append(byPath[path], int64(duration))
				got := nativeDownload(t, out)
				for i, v := range got {
					if math.Float32bits(v) != math.Float32bits(separate[i]) {
						t.Fatal("timing changed work/output", block, path, i)
					}
				}
				left()
				right()
			}
		}
		medians := map[string]float64{}
		for path, values := range byPath {
			sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
			medians[path] = float64(values[5]+values[6]) / 2
		}
		for _, s := range samples {
			data, _ := json.Marshal(s)
			t.Log("NATIVE_TIMING_SAMPLE " + string(data))
		}
		data, _ := json.Marshal(map[string]any{"rows": rows, "width": width, "heads": heads, "stages": 6, "samples_per_path": 12, "median_ns": medians, "separate_over_plan": medians["separate"] / medians["plan"], "gpu_timestamps": false, "outputs_bit_exact": true, "warmup_per_path": 2})
		t.Log("NATIVE_TIMING " + string(data))
	}
}
