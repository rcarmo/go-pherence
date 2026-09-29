package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedVulkanStreamingWindowPyTorchParity(t *testing.T) {
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
	tower, e1 := LoadOfflineAudioTower(file)
	head, e2 := LoadOfflineHead(file)
	compressor, e3 := LoadSpeakerCompressor(file)
	e4 := file.Close()
	for _, e := range []error{e1, e2, e3, e4} {
		if e != nil {
			t.Fatal(e)
		}
	}
	cache, err := NewSpeakerCache(compressor)
	if err != nil {
		t.Fatal(err)
	}
	device, err := NewVulkanStreamingTower(tower)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	window := &StreamingWindow{Tower: tower, VulkanTower: device, Head: head, Cache: cache}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	before := vk.VulkanMemoryStats()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := window.ForwardPreparedContext(cancelled, stacked[:13*projectedWidth], 9, 4); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled prepared window: %v", err)
	}
	if device.tower != nil || len(cache.fifo) != 0 || len(cache.speaker) != 0 {
		t.Fatal("cancelled window mutated cache or allocated Vulkan tower")
	}
	for step := 0; step <= 10; step++ {
		chunk := stacked[step*9*projectedWidth : (step*9+13)*projectedWidth]
		_, logits, err := window.ForwardPreparedContext(context.Background(), chunk, 9, 4)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if step > 2 && step != 5 && step != 10 {
			continue
		}
		ref := readStackingFixture(t, fmt.Sprintf("testdata/jfk_stream_step%d_logits.f32.gz", step), len(logits))
		maxAbs, mean, outside := towerError(logits, ref)
		t.Logf("step=%d logits max_abs=%g mean_abs=%g outside=%d", step, maxAbs, mean, outside)
		if outside != 0 || mean > 1e-5 {
			t.Fatalf("step=%d logits differ from PyTorch", step)
		}
	}
	if _, err := device.Forward(cancelled, stacked[:13*projectedWidth], 13); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled resident forward: %v", err)
	}
	if err := device.Close(); err != nil {
		t.Fatal(err)
	}
	if err := device.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := device.Forward(context.Background(), stacked[:13*projectedWidth], 13); err == nil {
		t.Fatal("accepted forward after close")
	}
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain {
		t.Fatalf("Vulkan resources before=%+v after=%+v", before, after)
	}
}
