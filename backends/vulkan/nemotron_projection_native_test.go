package vulkan

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

// Opt-in native-device probe of the existing F32 linear shader as a mel
// projection. It measures transfers separately from resident dispatch; this
// is not a Vulkan implementation of the Nemotron FFT or either whole model.
func TestNemotronVulkanNativeMelProjection(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN=1 for authorised native qualification")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native qualification requires go test -timeout at most 2m")
	}
	folder := os.Getenv("GO_PHERENCE_NEMOTRON_VULKAN_FIXTURE_DIR")
	if folder == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_VULKAN_FIXTURE_DIR to independently generated power/filter tensors")
	}
	const rows, bins, mels = 1100, 257, 128
	read := func(name string, count int) []float32 {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(folder, name))
		if err != nil || len(b) != count*4 {
			t.Fatalf("%s: bytes=%d want=%d err=%v", name, len(b), count*4, err)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return out
	}
	power := read("power.f32", 1101*bins)[:rows*bins]
	filters := read("filters.f32", mels*bins)
	if !VulkanInit() {
		t.Fatal("Vulkan initialization failed")
	}
	device := VulkanDeviceName()
	if !strings.Contains(device, "NVIDIA GeForce RTX 3060") || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatalf("unexpected physical device %q; this gate is specific to RTX 3060", device)
	}
	before := VulkanMemoryStats()
	func() {
		ctx := context.Background()
		operator, err := NewVkLinearF32(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if e := operator.Close(); e != nil {
				t.Error(e)
			}
		}()
		arena, err := NewVkTensorArena(ctx, 4<<20)
		if err != nil {
			t.Fatal(err)
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
		x, w, bias, out := alloc(rows, bins), alloc(mels, bins), alloc(mels), alloc(rows, mels)
		zeros := make([]float32, mels)
		start := time.Now()
		for _, pair := range []struct {
			tensor *VkTensorF32
			data   []float32
		}{{x, power}, {w, filters}, {bias, zeros}} {
			if e := pair.tensor.Upload(ctx, pair.data); e != nil {
				t.Fatal(e)
			}
		}
		upload := time.Since(start)
		start = time.Now()
		if err := operator.Forward(ctx, out, x, w, bias); err != nil {
			t.Fatal(err)
		}
		first := time.Since(start)
		got := make([]float32, rows*mels)
		start = time.Now()
		if err := out.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		download := time.Since(start)
		// Independently generated PyTorch STFT/mel inputs provide a distinct
		// reference for the projection, while the full Go frontend is checked
		// separately against the independent PyTorch feature fixture.
		var maxAbs, sumAbs float64
		var outside int
		for r := 0; r < rows; r++ {
			for m := 0; m < mels; m++ {
				var want float64
				for k := 0; k < bins; k++ {
					want += float64(power[r*bins+k]) * float64(filters[m*bins+k])
				}
				v := float64(got[r*mels+m])
				d := math.Abs(v - want)
				if d > maxAbs {
					maxAbs = d
				}
				sumAbs += d
				if math.IsNaN(v) || math.IsInf(v, 0) || d > 2e-3+2e-4*math.Abs(want) {
					outside++
				}
			}
		}
		if outside != 0 || sumAbs/float64(len(got)) > 1e-4 {
			t.Fatalf("mel projection error max=%g mean=%g outside=%d", maxAbs, sumAbs/float64(len(got)), outside)
		}
		// Match CPU SIMD against the same independent STFT/filter inputs.
		// Weight layout for DenseNTTo is [mels,bins], matching the Vulkan tensor.
		cpuProjection := make([]float32, rows*mels)
		cpuTimes := make([]time.Duration, 5)
		for i := range cpuTimes {
			clear(cpuProjection)
			start = time.Now()
			if !simd.DenseNTTo(cpuProjection, power, filters, rows, mels, bins, 1, bins, bins, mels) {
				t.Fatal("CPU SIMD projection rejected checked input")
			}
			cpuTimes[i] = time.Since(start)
		}
		var cpuMaxAbs float64
		for i, value := range got {
			cpuMaxAbs = math.Max(cpuMaxAbs, math.Abs(float64(value-cpuProjection[i])))
		}
		if cpuMaxAbs > 2e-3 {
			t.Fatalf("Vulkan/CPU projection drift max=%g", cpuMaxAbs)
		}
		// Repeat resident dispatch with all inputs and output in the arena.
		times := make([]time.Duration, 5)
		for i := range times {
			start = time.Now()
			if err := operator.Forward(ctx, out, x, w, bias); err != nil {
				t.Fatal(err)
			}
			times[i] = time.Since(start)
		}
		for i := range got {
			got[i] = 0
		}
		if err := out.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(float64(got[0])) || math.IsInf(float64(got[0]), 0) {
			t.Fatal("non-finite resident output")
		}
		// Measure CPU SIMD frontend separately in this process, excluding WAV I/O.
		pcm, rate, err := audio.WAV("../../testdata/jfk.wav")
		if err != nil || rate != 16000 {
			t.Fatalf("WAV rate=%d err=%v", rate, err)
		}
		start = time.Now()
		_, frames, err := audio.NemotronLogMel(pcm)
		cpu := time.Since(start)
		if err != nil || frames != 1101 {
			t.Fatalf("CPU frontend frames=%d err=%v", frames, err)
		}
		t.Logf("NEMOTRON_VULKAN_PROJECTION device=%q rows=%d bins=%d mels=%d max_abs=%g mean_abs=%g cpu_max_abs=%g upload=%s first_dispatch=%s download=%s resident_dispatch=%v cpu_simd_projection=%v cpu_full_frontend=%s", device, rows, bins, mels, maxAbs, sumAbs/float64(len(got)), cpuMaxAbs, upload, first, download, times, cpuTimes, cpu)
		if upload+first+download <= 0 {
			t.Fatal(fmt.Errorf("invalid timing sample"))
		}
	}()
	after := VulkanMemoryStats()
	if before.Allocations != after.Allocations || before.Bytes != after.Bytes {
		t.Fatalf("Vulkan allocation leak: before=%+v after=%+v", before, after)
	}
}
