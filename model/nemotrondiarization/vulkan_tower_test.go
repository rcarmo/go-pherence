package nemotrondiarization

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedVulkanAudioTowerPyTorchParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_TOWER") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_TOWER=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	if !vk.VulkanInit() {
		t.Skip("Vulkan unavailable")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadOfflineAudioTower(file)
	closeErr := file.Close()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	const rows = 138
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", rows*projectedWidth)
	before := vk.VulkanMemoryStats()
	ctx := context.Background()
	tower, err := NewVulkanAudioTower(ctx, model, rows)
	if err != nil {
		if tower != nil {
			defer tower.Close()
		}
		t.Fatal(err)
	}
	defer tower.Close()
	got, runErr := tower.forward(ctx, input, func(index int, tensor *vk.VkTensorF32) error {
		switch index {
		case 2, 7, 15, 23, 30:
			values := make([]float32, rows*projectedWidth)
			if err := tensor.Download(ctx, values); err != nil {
				return err
			}
			name := fmt.Sprintf("layer%d_complete", index)
			ref := readStackingFixture(t, "testdata/jfk_full_"+name+".f32.gz", len(values))
			maxAbs, mean, outside := towerError(values, ref)
			t.Logf("%s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
			limit := 2e-5
			if index == 30 {
				limit = 3e-5
			}
			if outside != 0 || mean > limit {
				return fmt.Errorf("%s differs from pinned PyTorch", name)
			}
		}
		return nil
	})
	if runErr != nil {
		t.Fatal(runErr)
	}
	ref := readStackingFixture(t, "testdata/jfk_full_tower_normal.f32.gz", len(got))
	maxAbs, mean, outside := towerError(got, ref)
	t.Logf("tower_normal max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatal("resident Vulkan tower differs from pinned PyTorch")
	}
	second, err := tower.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	maxAbs, mean, outside = towerError(second, ref)
	t.Logf("production path repeat max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatal("production tower differs from PyTorch")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := tower.Forward(cancelled, input); err != context.Canceled {
		t.Fatalf("cancelled tower run: %v", err)
	}
	if err := tower.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tower.Close(); err != nil {
		t.Fatal(err)
	}
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain {
		t.Fatalf("Vulkan resources before=%+v after=%+v", before, after)
	}
}

// BenchmarkReleasedAudioTower includes layer 0, all 30 layers, final norm and
// GPU input/output transfers. GPU setup and teardown are reported separately.
func BenchmarkReleasedAudioTower(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	model, loadErr := LoadOfflineAudioTower(file)
	closeErr := file.Close()
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	if closeErr != nil {
		b.Fatal(closeErr)
	}
	for _, rows := range []int{16, 138} {
		b.Run(fmt.Sprintf("rows=%d/simd", rows), func(b *testing.B) {
			input := benchmarkTowerInput(b, rows)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := model.ForwardOffline(input, rows); err != nil {
					b.Fatal(err)
				}
			}
		})
		if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_TOWER") != "1" || !vk.VulkanInit() {
			continue
		}
		b.Run(fmt.Sprintf("rows=%d/vulkan", rows), func(b *testing.B) {
			input := benchmarkTowerInput(b, rows)
			ctx := context.Background()
			setup := time.Now()
			tower, err := NewVulkanAudioTower(ctx, model, rows)
			if err != nil {
				if tower != nil {
					tower.Close()
				}
				b.Fatal(err)
			}
			setupTime := time.Since(setup)
			defer tower.Close()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := tower.Forward(ctx, input); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			closeStart := time.Now()
			if err := tower.Close(); err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(float64(setupTime.Microseconds()), "setup-us")
			b.ReportMetric(float64(time.Since(closeStart).Microseconds()), "close-us")
		})
	}
}

func benchmarkTowerInput(b *testing.B, rows int) []float32 {
	b.Helper()
	// The checked-in independent fixture is 138 rows; a prefix is a valid
	// unmasked window but not a labelled diarization example.
	f, err := os.Open("testdata/jfk_stacking_transformers_5_18.f32.gz")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		b.Fatal(err)
	}
	defer gz.Close()
	data := make([]byte, 138*projectedWidth*4)
	if _, err := io.ReadFull(gz, data); err != nil {
		b.Fatal(err)
	}
	out := make([]float32, rows*projectedWidth)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func TestVulkanAudioTowerRejectsInvalidAndCleansFailedSetup(t *testing.T) {
	if _, err := NewVulkanAudioTower(context.Background(), nil, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := (*VulkanAudioTower)(nil).Forward(context.Background(), make([]float32, projectedWidth)); err == nil {
		t.Fatal("accepted nil tower")
	}
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_TOWER") != "1" || !vk.VulkanInit() {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_TOWER=1 and provide Vulkan")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadOfflineAudioTower(file)
	closeErr := file.Close()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewVulkanAudioTower(ctx, model, 138); err != context.Canceled {
		t.Fatalf("cancelled constructor: %v", err)
	}
	before := vk.VulkanMemoryStats()
	// Fail after the first 32 MiB layer arena was admitted, to exercise
	// constructor teardown across partially built GPU resources.
	budget := vk.VulkanMemoryBudget{MaxBytes: before.Bytes + 40<<20}
	if err := vk.VulkanSetMemoryBudget(budget); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	}()
	failed, err := NewVulkanAudioTower(context.Background(), model, 138)
	if err == nil {
		if failed != nil {
			failed.Close()
		}
		t.Fatal("accepted insufficient Vulkan memory budget")
	}
	if failed != nil {
		if closeErr := failed.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	after := vk.VulkanMemoryStats()
	if before.Bytes != after.Bytes || before.Allocations != after.Allocations || after.InFlight || after.Uncertain {
		t.Fatalf("failed setup leaked resources: before=%+v after=%+v", before, after)
	}
}

func towerError(got, ref []float32) (maxAbs, mean float64, outside int) {
	if len(got) != len(ref) || len(ref) == 0 {
		return math.Inf(1), math.Inf(1), 1
	}
	for i, value := range got {
		delta := math.Abs(float64(value - ref[i]))
		maxAbs = math.Max(maxAbs, delta)
		mean += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	return maxAbs, mean / float64(len(got)), outside
}
