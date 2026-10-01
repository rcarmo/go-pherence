package whisper

import (
	"context"
	"errors"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanEncoderScoreILPNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_ENCODER_SCOREILP") != "1" {
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
		candidate, err := NewVulkanEncoderTile64Key32ScoreILP(ctx, source, c.MaxLength)
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
					if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) || math.Abs(float64(out[i]-ref[i])) > 1e-4+1e-4*math.Abs(float64(ref[i])) {
						t.Fatalf("changed arithmetic i=%d baseline=%g candidate=%g", i, ref[i], out[i])
					}
				}
			}
			// Mid-request cancellation may retain a native submission. Drain
			// only that owner before reuse; never hide failure with CPU fallback.
			counter := newCheckpointContext(0)
			defer counter.cancel()
			if _, err := candidate.Forward(counter, mel); err != nil {
				t.Fatal(err)
			}
			for _, at := range []int{1, counter.calls / 2, counter.calls - 4} {
				fault := newCheckpointContext(at)
				_, err := candidate.Forward(fault, mel)
				fault.cancel()
				if !errors.Is(err, context.Canceled) {
					t.Fatal("candidate cancellation", at, err)
				}
				if !vk.VulkanReady() {
					drainCtx, stop := context.WithTimeout(context.Background(), time.Second)
					err := vk.VulkanDrain(drainCtx, time.Second)
					stop()
					if err != nil {
						t.Fatal(err)
					}
				}
				out, err := candidate.Forward(ctx, mel)
				if err != nil {
					t.Fatal("cancel retry", err)
				}
				for i := range out {
					if math.IsNaN(float64(out[i])) || math.Abs(float64(out[i]-ref[i])) > 1e-4+1e-4*math.Abs(float64(ref[i])) {
						t.Fatal("cancel retry drift", i)
					}
				}
			}
			refs := vulkanScalarBoundaries(source, mel, c.MaxLength)
			out, err := candidate.Forward(ctx, mel)
			if err != nil {
				t.Fatal(err)
			}
			nativeEncoderCompare(t, "key32-independent", out, refs[len(refs)-1])
		}()
	}
}
