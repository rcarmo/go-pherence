package nemotrondiarization

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedVulkanAudioLayerPyTorchParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_VULKAN_LAYER") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_VULKAN_LAYER=1")
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
	first, err := LoadLayer0Complete(file)
	if err != nil {
		file.Close()
		t.Fatal(err)
	}
	layer, err := LoadLayer1Complete(file)
	if err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	ctx := context.Background()
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			input, err := first.ForwardOffline(stacked[:rows*projectedWidth], rows)
			if err != nil {
				t.Fatal(err)
			}
			before := vk.VulkanMemoryStats()
			op, err := NewVulkanAudioLayer(ctx, layer, rows)
			if err != nil {
				if op != nil {
					defer op.Close()
				}
				t.Fatal(err)
			}
			defer op.Close() // retain cleanup on an early assertion failure
			got, forwardErr := op.Forward(ctx, input)
			closeErr := op.Close()
			if forwardErr != nil {
				t.Fatal(forwardErr)
			}
			if closeErr != nil {
				t.Fatal(closeErr)
			}
			after := vk.VulkanMemoryStats()
			if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain {
				t.Fatalf("Vulkan resources before=%+v after=%+v", before, after)
			}
			name := "testdata/jfk_layer1_complete.f32.gz"
			if rows == 138 {
				name = "testdata/jfk_full_layer1_complete.f32.gz"
			}
			ref := readStackingFixture(t, name, rows*projectedWidth)
			var maxAbs, sumAbs float64
			var outside int
			for i, v := range got {
				d := math.Abs(float64(v - ref[i]))
				maxAbs = math.Max(maxAbs, d)
				sumAbs += d
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(got))
			t.Logf("rows=%d max_abs=%g mean_abs=%g outside=%d", rows, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatal("resident Vulkan layer differs from pinned PyTorch")
			}
		})
	}
}
