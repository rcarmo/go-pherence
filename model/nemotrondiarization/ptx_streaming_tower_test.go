package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPTXStreamingWindowPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	if !ptx.SgemmReady() {
		t.Skip("CUDA unavailable")
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
	device, err := NewPTXStreamingTower(tower)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	window := &StreamingWindow{Tower: tower, PTXTower: device, Head: head, Cache: cache}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := window.ForwardPreparedContext(cancelled, stacked[:13*projectedWidth], 9, 4); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled prepared window: %v", err)
	}
	if device.tower != nil || len(cache.fifo) != 0 || len(cache.speaker) != 0 {
		t.Fatal("cancelled window mutated cache or allocated PTX tower")
	}
	for step := 0; step <= 10; step++ {
		previous := device.tower
		chunk := stacked[step*9*projectedWidth : (step*9+13)*projectedWidth]
		_, logits, err := window.ForwardPreparedContext(context.Background(), chunk, 9, 4)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if device.tower == nil || device.tower.maxRows != maxPreparedDiarizationRows || (previous != nil && device.tower != previous) {
			t.Fatalf("step=%d rebuilt resident PTX tower or changed bounded capacity", step)
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
	device.Close()
	device.Close()
	if _, err := device.Forward(context.Background(), stacked[:13*projectedWidth], 13); err == nil {
		t.Fatal("accepted forward after close")
	}
}

func TestPTXStreamingTowerSelectionRejectsConflicts(t *testing.T) {
	if _, err := NewPTXStreamingTower(nil); err == nil {
		t.Fatal("accepted nil model")
	}
	request := &PCMStreamingRequest{window: &StreamingWindow{Tower: &OfflineAudioTower{first: &Layer0Complete{}, remaining: make([]*Layer1Complete, 30)}, VulkanTower: &VulkanStreamingTower{}}}
	if err := request.EnablePTXTower(); err == nil {
		t.Fatal("accepted both GPU towers")
	}
	request.window.VulkanTower = nil
	if err := request.EnablePTXTower(); err != nil {
		t.Fatal(err)
	}
	if err := request.EnableVulkanTower(); err == nil {
		t.Fatal("accepted conflicting Vulkan tower")
	}
	if err := request.ClosePTXTower(); err != nil {
		t.Fatal(err)
	}
	if err := request.ClosePTXTower(); err != nil {
		t.Fatal(err)
	}
}
