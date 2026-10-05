package whisper

import (
	"context"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanEncoderTile64Native(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_ENCODER_TILE64") != "1" {
		t.Skip("explicit candidate qualification")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer cancel()
	if !vk.VulkanInit() {
		t.Fatal("Vulkan init")
	}
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(vk.VulkanDeviceName(), want) || strings.Contains(strings.ToLower(vk.VulkanDeviceName()), "llvmpipe") {
		t.Fatal("physical device required")
	}
	before := vk.VulkanMemoryStats()
	defer func() {
		after := vk.VulkanMemoryStats()
		if before.Bytes != after.Bytes || before.Allocations != after.Allocations {
			t.Error("native leak", before, after)
		}
	}()
	for _, c := range []Config{{EncoderDModel: 8, EncoderLayers: 2, EncoderHeads: 2, EncoderFFNDim: 17, HeadDim: 4, NumMelBins: 3, MaxLength: 34}, {EncoderDModel: 64, EncoderLayers: 1, EncoderHeads: 2, EncoderFFNDim: 129, HeadDim: 32, NumMelBins: 5, MaxLength: 130}} {
		source := vulkanToyEncoder(t, c)
		mel := make([]float32, c.NumMelBins*c.MaxLength)
		for i := range mel {
			mel[i] = float32(math.Sin(float64(i) * .13))
		}
		candidate, err := NewVulkanEncoderRegTile64(ctx, source, c.MaxLength)
		if err != nil {
			if candidate != nil {
				candidate.Close()
			}
			t.Fatal(err)
		}
		baseline, err := NewVulkanEncoder(ctx, source, c.MaxLength)
		if err != nil {
			candidate.Close()
			if baseline != nil {
				baseline.Close()
			}
			t.Fatal(err)
		}
		func() {
			defer func() {
				if err := candidate.Close(); err != nil {
					t.Error("candidate close", err)
				}
			}()
			defer func() {
				if err := baseline.Close(); err != nil {
					t.Error("baseline close", err)
				}
			}()
			ref, err := baseline.Forward(ctx, mel)
			if err != nil {
				t.Fatal(err)
			}
			for run := 0; run < 3; run++ {
				out, err := candidate.Forward(ctx, mel)
				if err != nil {
					t.Fatal(err)
				}
				if len(out) != len(ref) {
					t.Fatal("output geometry")
				}
				for i := range out {
					if out[i] != ref[i] {
						t.Fatalf("changed arithmetic i=%d baseline=%g candidate=%g", i, ref[i], out[i])
					}
				}
			}
			refs := vulkanScalarBoundaries(source, mel, c.MaxLength)
			nativeEncoderCompare(t, "tile64-independent", ref, refs[len(refs)-1])
		}()
	}
}
