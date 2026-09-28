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
)

// Native probe of existing Vulkan F32 linear on pinned ASR causal-stem patches.
// CPU patch extraction and channel-major output reordering are measured too;
// this test does not implement a Vulkan causal convolution or full encoder.
func TestNemotronVulkanNativeASRStemProjection(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_STEM") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_STEM=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_STEM_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_STEM_FIXTURE_DIR")
	}
	const positions, k, outDim = 1105, 9, 256
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
	patches, weight, bias := read("patches.f32", positions*k), read("weight.f32", outDim*k), read("bias.f32", outDim)
	file, e := os.Open("../../model/nemotronasr/testdata/output.f32.gz")
	if e != nil {
		t.Fatal(e)
	}
	gz, e := gzip.NewReader(file)
	if e != nil {
		file.Close()
		t.Fatal(e)
	}
	ref, e := io.ReadAll(io.LimitReader(gz, positions*outDim*4+1))
	gz.Close()
	file.Close()
	if e != nil || len(ref) != positions*outDim*4 {
		t.Fatalf("reference bytes=%d err=%v", len(ref), e)
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
		op, e := NewVkLinearF32(ctx)
		if e != nil {
			t.Fatal(e)
		}
		defer func() {
			if e := op.Close(); e != nil {
				t.Error(e)
			}
		}()
		arena, e := NewVkTensorArena(ctx, 3<<20)
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
		x, w, b, y := alloc(positions, k), alloc(outDim, k), alloc(outDim), alloc(positions, outDim)
		start := time.Now()
		for _, p := range []struct {
			v    *VkTensorF32
			data []float32
		}{{x, patches}, {w, weight}, {b, bias}} {
			if e := p.v.Upload(ctx, p.data); e != nil {
				t.Fatal(e)
			}
		}
		upload := time.Since(start)
		start = time.Now()
		if e := op.Forward(ctx, y, x, w, b); e != nil {
			t.Fatal(e)
		}
		first := time.Since(start)
		projected := make([]float32, positions*outDim)
		start = time.Now()
		if e := y.Download(ctx, projected); e != nil {
			t.Fatal(e)
		}
		download := time.Since(start)
		result := make([]float32, len(projected))
		start = time.Now()
		for ch := 0; ch < outDim; ch++ {
			for pos := 0; pos < positions; pos++ {
				result[ch*positions+pos] = projected[pos*outDim+ch]
			}
		}
		reorder := time.Since(start)
		var maxAbs, sumAbs, float64Max, float64Sum float64
		var outside, differing int
		for i, v := range result {
			want := math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
			ch, pos := i/positions, i%positions
			var independent float64 = float64(bias[ch])
			for j := 0; j < k; j++ {
				independent += float64(patches[pos*k+j]) * float64(weight[ch*k+j])
			}
			refError := math.Abs(float64(v) - independent)
			float64Max = math.Max(float64Max, refError)
			float64Sum += refError
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || refError > 3e-5+2e-5*math.Abs(independent) {
				outside++
			}
			d := math.Abs(float64(v - want))
			if d != 0 {
				differing++
			}
			if d > maxAbs {
				maxAbs = d
			}
			sumAbs += d
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-5+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		mean := sumAbs / float64(len(result))
		if outside != 0 || mean > 3e-6 || float64Sum/float64(len(result)) > 3e-6 {

			t.Fatalf("Vulkan ASR stem max=%g mean=%g outside=%d", maxAbs, mean, outside)
		}
		resident := make([]time.Duration, 5)
		for i := range resident {
			start = time.Now()
			if e := op.Forward(ctx, y, x, w, b); e != nil {
				t.Fatal(e)
			}
			resident[i] = time.Since(start)
		}
		t.Logf("NEMOTRON_VULKAN_ASR_STEM device=%q max_abs=%g mean_abs=%g float64_max=%g float64_mean=%g differing=%d upload=%s first_dispatch=%s download=%s reorder=%s resident=%v", device, maxAbs, mean, float64Max, float64Sum/float64(len(result)), differing, upload, first, download, reorder, resident)
	}()
	after := VulkanMemoryStats()
	if before.Allocations != after.Allocations || before.Bytes != after.Bytes {
		t.Fatalf("Vulkan leak before=%+v after=%+v", before, after)
	}
}
