package community1

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// Explicit trained full-pipeline native gate. It is one bounded 30-second
// public sample and never runs in ordinary tests or selects a service default.
func TestVulkanCommunity1TrainedDiarization(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_COMMUNITY_DIARIZATION") != "1" {
		t.Skip("set GO_PHERENCE_TEST_VULKAN_COMMUNITY_DIARIZATION=1 in an authorised compute window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 10*time.Minute {
		t.Fatal("trained Community-1 Vulkan qualification requires go test -timeout of at most 10m")
	}
	if !vk.VulkanInit() {
		t.Fatal("native Vulkan initialisation failed")
	}
	device := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	lower := strings.ToLower(device)
	if want == "" || !strings.Contains(device, want) || strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("set GO_PHERENCE_VULKAN_DEVICE to the expected physical device", device)
	}
	ctx := context.Background()
	segmentation, filters, embedding, prepared := loadTrainedDiarizationModels(t, ctx)
	reader := openTrainedDiarizationPCM(t, ctx)
	before := vk.VulkanMemoryStats()
	started := time.Now()
	model, err := NewVulkanDiarization(ctx, segmentation, filters, embedding, prepared.Model, 160000)
	if err != nil {
		t.Fatal(err)
	}
	constructed := time.Since(started)
	stats := model.Stats()
	live := vk.VulkanMemoryStats()
	if stats.WindowSamples != 160000 || stats.SegmentationFrames != 589 || stats.FbankFrames != 998 || stats.LocalSpeakers != 3 || stats.EmbeddingDimension != 256 {
		nativeCommunityClose(t, model)
		t.Fatal("trained Vulkan owner dimensions", stats)
	}
	cfg := DiarizationPCMConfig{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: LowestIndexTies}
	runStarted := time.Now()
	result, err := model.RunPCM(ctx, reader, 480000, cfg, SincNetSIMDFMA, HeadSIMD)
	runElapsed := time.Since(runStarted)
	closeStarted := time.Now()
	nativeCommunityClose(t, model)
	closeElapsed := time.Since(closeStarted)
	after := vk.VulkanMemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Windows) != 21 || result.Grid.Frames != 589 || result.LocalSpeakers != 3 || result.EmbeddingDimension != 256 || result.Postprocess.Path != "clustered" || result.Postprocess.TrainingRows != 37 || result.Postprocess.Clusters != 2 || len(result.Postprocess.FullTurns) != 13 || len(result.Postprocess.ExclusiveTurns) != 12 || len(result.Postprocess.Timeline.AmbiguousFrames) != 84 {
		t.Fatal("trained Vulkan result contract", result.Postprocess.Path, result.Postprocess.TrainingRows, result.Postprocess.Clusters, len(result.Postprocess.FullTurns), len(result.Postprocess.ExclusiveTurns), len(result.Postprocess.Timeline.AmbiguousFrames))
	}
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain || after.DeviceLost {
		t.Fatalf("trained Community-1 Vulkan allocation/state leak: before%+v live%+v after%+v", before, live, after)
	}
	t.Logf("TRAINED_VULKAN_DIARIZATION device=%q construct=%s run=%s close=%s live_bytes=%d live_allocations=%d", device, constructed, runElapsed, closeElapsed, live.Bytes-before.Bytes, live.Allocations-before.Allocations)
	out := os.Getenv("GO_PHERENCE_VULKAN_COMMUNITY_DIARIZATION_OUTPUT")
	if out == "" {
		return
	}
	data, err := json.MarshalIndent(struct {
		Config DiarizationPCMConfig
		Result *DiarizationPCMResult
	}{cfg, result}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write(append(data, '\n'))
	closeErr := file.Close()
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
}
