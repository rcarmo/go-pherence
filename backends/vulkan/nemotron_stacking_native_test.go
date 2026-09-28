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

// Real-device operator gate: pinned diarization feature stacking [138,1024]
// times released [512,1024] weight. Transformer/head and complete frontend
// remain separate unqualified stages.
func TestNemotronVulkanNativeDiarizationStacking(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_STACKING") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_STACKING=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_STACKING_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_STACKING_FIXTURE_DIR")
	}
	const rows, in, outDim = 138, 1024, 512
	read := func(name string, n int) []float32 {
		t.Helper()
		raw, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil || len(raw) != n*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(raw), e)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	}
	features, weight := read("stacked.f32", rows*in), read("weight.f32", outDim*in)
	fixture, err := os.Open("../../model/nemotrondiarization/testdata/jfk_stacking_transformers_5_18.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(fixture)
	if err != nil {
		fixture.Close()
		t.Fatal(err)
	}
	compressed, err := io.ReadAll(io.LimitReader(gz, rows*outDim*4+1))
	gz.Close()
	fixture.Close()
	if err != nil || len(compressed) != rows*outDim*4 {
		t.Fatalf("reference size=%d err=%v", len(compressed), err)
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
		operator, e := NewVkLinearF32(ctx)
		if e != nil {
			t.Fatal(e)
		}
		defer func() {
			if e := operator.Close(); e != nil {
				t.Error(e)
			}
		}()
		arena, e := NewVkTensorArena(ctx, 5<<20)
		if e != nil {
			t.Fatal(e)
		}
		defer func() {
			if e := arena.Close(); e != nil {
				t.Error(e)
			}
		}()
		alloc := func(shape ...int) *VkTensorF32 {
			t.Helper()
			v, e := arena.AllocF32(ctx, shape...)
			if e != nil {
				t.Fatal(e)
			}
			return v
		}
		x, w, bias, out := alloc(rows, in), alloc(outDim, in), alloc(outDim), alloc(rows, outDim)
		start := time.Now()
		for _, p := range []struct {
			v    *VkTensorF32
			data []float32
		}{{x, features}, {w, weight}, {bias, make([]float32, outDim)}} {
			if e := p.v.Upload(ctx, p.data); e != nil {
				t.Fatal(e)
			}
		}
		upload := time.Since(start)
		start = time.Now()
		if e := operator.Forward(ctx, out, x, w, bias); e != nil {
			t.Fatal(e)
		}
		first := time.Since(start)
		got := make([]float32, rows*outDim)
		start = time.Now()
		if e := out.Download(ctx, got); e != nil {
			t.Fatal(e)
		}
		download := time.Since(start)
		var maxAbs, sumAbs float64
		var outside int
		for i, v := range got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(compressed[i*4:]))
			d := math.Abs(float64(v - want))
			if d > maxAbs {
				maxAbs = d
			}
			sumAbs += d
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 2e-3+2e-4*math.Abs(float64(want)) {
				outside++
			}
		}
		mean := sumAbs / float64(len(got))
		if outside != 0 || mean > 1e-4 {
			t.Fatalf("Vulkan stack max=%g mean=%g outside=%d", maxAbs, mean, outside)
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
		for i, v := range got {
			cpuMax = math.Max(cpuMax, math.Abs(float64(v-cpu[i])))
		}
		if cpuMax > 2e-3 {
			t.Fatalf("Vulkan/CPU drift=%g", cpuMax)
		}
		resident := make([]time.Duration, 5)
		for i := range resident {
			start = time.Now()
			if e := operator.Forward(ctx, out, x, w, bias); e != nil {
				t.Fatal(e)
			}
			resident[i] = time.Since(start)
		}
		if e := out.Download(ctx, got); e != nil {
			t.Fatal(e)
		}
		for _, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("non-finite resident output")
			}
		}
		t.Logf("NEMOTRON_VULKAN_STACKING device=%q max_abs=%g mean_abs=%g cpu_max_abs=%g upload=%s first_dispatch=%s download=%s resident_dispatch=%v cpu_simd=%v", device, maxAbs, mean, cpuMax, upload, first, download, resident, cpuTimes)
	}()
	after := VulkanMemoryStats()
	if before.Allocations != after.Allocations || before.Bytes != after.Bytes {
		t.Fatalf("Vulkan leak before=%+v after=%+v", before, after)
	}
}
