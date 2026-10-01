package whisper

import (
	"context"
	"fmt"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Construction attribution only: same file, model, original-value admission and
// packed allocation. Each owner closes before the next arm. No inference times.
func TestVulkanQ5PreparationComparison(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_Q5_PREPARATION") != "1" {
		t.Skip("explicit preparation window")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 300*time.Second {
		t.Fatal("bounded300s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	model, _, _, file := pinnedLegacyWhisperModelOpen(t, ctx)
	defer file.Close()
	if !vk.VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("device")
	}
	beforeNative := vk.VulkanMemoryStats()
	var shapes []vk.VkLinearQ5Shape
	var names []string
	var values [][]float32
	for layer := range model.Encoder.Layers {
		l := &model.Encoder.Layers[layer]
		shapes = append(shapes, vk.VkLinearQ5Shape{OutDim: model.Config.EncoderFFNDim, InDim: model.Config.EncoderDModel}, vk.VkLinearQ5Shape{OutDim: model.Config.EncoderDModel, InDim: model.Config.EncoderFFNDim})
		names = append(names, fmt.Sprintf("encoder.blocks.%d.mlp.0.weight", layer), fmt.Sprintf("encoder.blocks.%d.mlp.2.weight", layer))
		values = append(values, l.FC1Weight, l.FC2Weight)
	}
	read := func(ctx context.Context, index int) ([]byte, error) {
		raw, shape, err := file.Q5Blocks(ctx, names[index])
		if err != nil {
			return nil, err
		}
		if len(shape) != 2 || shape[0] != shapes[index].InDim || shape[1] != shapes[index].OutDim {
			return nil, fmt.Errorf("shape")
		}
		if err = checkOriginalQ5Values(ctx, raw, values[index]); err != nil {
			return nil, err
		}
		return raw, nil
	}
	var retained uint64
	for block := 0; block < 5; block++ {
		order := []bool{false, true}
		if block%2 == 1 {
			order = []bool{true, false}
		}
		for _, stream := range order {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			start := time.Now()
			var owner *vk.VkLinearQ5Set
			var err error
			if stream {
				owner, err = vk.NewVkLinearQ5SetStream(ctx, shapes, read)
			} else {
				matrices := make([]vk.VkLinearQ5Matrix, len(shapes))
				for i, s := range shapes {
					raw, e := read(ctx, i)
					if e != nil {
						t.Fatal(e)
					}
					matrices[i] = vk.VkLinearQ5Matrix{Blocks: raw, OutDim: s.OutDim, InDim: s.InDim}
				}
				owner, err = vk.NewVkLinearQ5Set(ctx, matrices)
			}
			elapsed := time.Since(start)
			runtime.ReadMemStats(&after)
			if err != nil {
				if owner != nil {
					owner.Close()
				}
				t.Fatal(err)
			}
			if retained == 0 {
				retained = owner.StorageBytes()
			} else if retained != owner.StorageBytes() {
				t.Fatal("extent")
			}
			if err = owner.Close(); err != nil {
				t.Fatal(err)
			}
			if got := vk.VulkanMemoryStats(); got.Bytes != beforeNative.Bytes || got.Allocations != beforeNative.Allocations {
				t.Fatal("native leak", got)
			}
			t.Logf("Q5_PREP_SAMPLE stream=%t block=%d ns=%d allocated_bytes=%d allocations=%d retained_native=%d", stream, block, elapsed.Nanoseconds(), after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs, retained)
		}
	}
	// Keep caller model live across every arm; this is not a CPU-widening removal.
	runtime.KeepAlive(model)
}
