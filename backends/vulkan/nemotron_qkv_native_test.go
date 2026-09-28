package vulkan

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Prepared first-layer pre-RoPE Q/K/V GEMM only: normalized input and weight
// are pinned separately. LayerNorm, RoPE, attention and the speaker head do not
// run on Vulkan here.
func TestNemotronVulkanNativeDiarizationQKV(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_QKV") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_QKV=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_QKV_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_QKV_FIXTURE_DIR")
	}
	const rows, in, outDim = 16, 512, 1536
	read := func(name string, n int) []float32 {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || len(raw) != n*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(raw), err)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	}
	features, weight := read("normal.f32", rows*in), read("weight.f32", outDim*in)
	reference := make([]float32, rows*outDim)
	for part, name := range []string{"q", "k", "v"} {
		file, err := os.Open("../../model/nemotrondiarization/testdata/jfk_layer0_" + name + ".f32.gz")
		if err != nil {
			t.Fatal(err)
		}
		gz, err := gzip.NewReader(file)
		if err != nil {
			file.Close()
			t.Fatal(err)
		}
		raw, err := io.ReadAll(io.LimitReader(gz, rows*in*4+1))
		gz.Close()
		file.Close()
		if err != nil || len(raw) != rows*in*4 {
			t.Fatalf("reference %s bytes=%d err=%v", name, len(raw), err)
		}
		for row := 0; row < rows; row++ {
			for j := 0; j < in; j++ {
				reference[row*outDim+part*in+j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[(row*in+j)*4:]))
			}
		}
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := VulkanDeviceName()
	if !strings.Contains(device, "NVIDIA GeForce RTX 3060") || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatalf("unexpected Vulkan device %q", device)
	}
	before := VulkanMemoryStats()
	func() {
		ctx := context.Background()
		op, err := NewVkLinearF32(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := op.Close(); err != nil {
				t.Error(err)
			}
		}()
		arena, err := NewVkTensorArena(ctx, 5<<20)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := arena.Close(); err != nil {
				t.Error(err)
			}
		}()
		alloc := func(shape ...int) *VkTensorF32 {
			t.Helper()
			value, err := arena.AllocF32(ctx, shape...)
			if err != nil {
				t.Fatal(err)
			}
			return value
		}
		x, w, bias, out := alloc(rows, in), alloc(outDim, in), alloc(outDim), alloc(rows, outDim)
		start := time.Now()
		for _, item := range []struct {
			value *VkTensorF32
			data  []float32
		}{{x, features}, {w, weight}, {bias, make([]float32, outDim)}} {
			if err := item.value.Upload(ctx, item.data); err != nil {
				t.Fatal(err)
			}
		}
		upload := time.Since(start)
		start = time.Now()
		if err := op.Forward(ctx, out, x, w, bias); err != nil {
			t.Fatal(err)
		}
		first := time.Since(start)
		got := make([]float32, len(reference))
		start = time.Now()
		if err := out.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		download := time.Since(start)
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range got {
			delta := math.Abs(float64(value - reference[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(reference[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(got))
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("Vulkan QKV max=%g mean=%g outside=%d", maxAbs, mean, outside)
		}
		cpu := make([]float32, len(got))
		cpuTimes := make([]time.Duration, 5)
		for i := range cpuTimes {
			clear(cpu)
			start = time.Now()
			if !simd.DenseNTTo(cpu, features, weight, rows, outDim, in, 1, in, in, outDim) {
				t.Fatal("CPU SIMD rejected shape")
			}
			cpuTimes[i] = time.Since(start)
		}
		var cpuMax float64
		for i, value := range got {
			cpuMax = math.Max(cpuMax, math.Abs(float64(value-cpu[i])))
		}
		if cpuMax > 3e-4 {
			t.Fatalf("Vulkan/CPU QKV drift=%g", cpuMax)
		}
		resident := make([]time.Duration, 5)
		for i := range resident {
			start = time.Now()
			if err := op.Forward(ctx, out, x, w, bias); err != nil {
				t.Fatal(err)
			}
			resident[i] = time.Since(start)
		}
		if err := out.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		for _, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite resident output")
			}
		}
		t.Logf("NEMOTRON_VULKAN_QKV device=%q max_abs=%g mean_abs=%g cpu_max_abs=%g upload=%s first_dispatch=%s download=%s resident_dispatch=%v cpu_simd=%v", device, maxAbs, mean, cpuMax, upload, first, download, resident, cpuTimes)
	}()
	after := VulkanMemoryStats()
	if before.Allocations != after.Allocations || before.Bytes != after.Bytes {
		t.Fatalf("Vulkan leak before=%+v after=%+v", before, after)
	}
}
