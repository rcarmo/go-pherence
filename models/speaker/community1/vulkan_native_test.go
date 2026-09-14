package community1

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// Explicit synthetic native-device gate. Run in an isolated process only after
// compute admission; ordinary tests never initialize Vulkan.
func TestVulkanCommunityNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_COMMUNITY") != "1" {
		t.Skip("set GO_PHERENCE_TEST_VULKAN_COMMUNITY=1 in an authorised compute window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 5*time.Minute {
		t.Fatal("Community-1 native qualification requires go test -timeout of at most 5m")
	}
	if !vk.VulkanInit() {
		t.Fatal("native Vulkan initialisation failed")
	}
	device := vk.VulkanDeviceName()
	if want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE"); want == "" || !strings.Contains(device, want) {
		t.Fatal("set GO_PHERENCE_VULKAN_DEVICE to the expected hardware device", device)
	}
	lower := strings.ToLower(device)
	if strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("software Vulkan is not native qualification", device)
	}
	before := vk.VulkanMemoryStats()
	t.Run("basic-block", nativeCommunityBlock)
	nativeCommunityMemory(t, before)
	t.Run("resnet-hybrid", nativeCommunityResNet)
	nativeCommunityMemory(t, before)
	t.Run("lstm", nativeCommunityLSTM)
	nativeCommunityMemory(t, before)
	t.Run("segmentation-features", nativeCommunitySegmentationFeatures)
	nativeCommunityMemory(t, before)
	t.Run("segmentation-pcm", nativeCommunitySegmentationPCM)
	nativeCommunityMemory(t, before)
}

func nativeCommunityMemory(t *testing.T, before vk.VulkanMemoryUsage) {
	t.Helper()
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain || after.DeviceLost {
		t.Fatalf("Community-1 Vulkan allocation/state leak: before%+v after%+v", before, after)
	}
}

func nativeCommunityClose(t *testing.T, closer interface{ Close() error }) {
	t.Helper()
	if err := closer.Close(); err != nil {
		if drainErr := vk.VulkanDrain(context.Background(), 5*time.Second); drainErr != nil {
			t.Fatalf("Community-1 Vulkan drain after close: %v / %v", err, drainErr)
		}
		if err := closer.Close(); err != nil {
			t.Fatal("Community-1 Vulkan close retry", err)
		}
	}
}

func nativeCommunityCompare(t *testing.T, name string, got, want []float32, absolute, relative float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d want %d", name, len(got), len(want))
	}
	maximum := 0.0
	for i := range got {
		err := math.Abs(float64(got[i]) - float64(want[i]))
		maximum = math.Max(maximum, err)
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || err > absolute+relative*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d] got%.9g want%.9g error%.6g", name, i, got[i], want[i], err)
		}
	}
	t.Logf("%s values=%d max_abs=%g", name, len(got), maximum)
}

func nativeCommunityBlock(t *testing.T) {
	fixture := loadBlockFixtures(t)[1]
	cpu, err := NewWeSpeakerBasicBlock(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	want, wantShape, err := cpu.Forward(context.Background(), fixture.Input, fixture.Shape, WeSpeakerBlockGEMM)
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := NewVulkanBasicBlock(context.Background(), cpu, fixture.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeCommunityClose(t, gpu)
	got, gotShape, err := gpu.Forward(context.Background(), fixture.Input)
	if err != nil || gotShape != wantShape {
		t.Fatal(gotShape, wantShape, err)
	}
	nativeCommunityCompare(t, "basic-block", got, want, 2e-4, 2e-4)
}

func nativeCommunityResNet(t *testing.T) {
	fixture := loadResNetFixtures(t)[0]
	cpu, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	wantFeatures, wantShape, err := cpu.ForwardFrames(context.Background(), fixture.Input, fixture.Frames, WeSpeakerBlockGEMM)
	if err != nil {
		t.Fatal(err)
	}
	trunk, err := NewVulkanResNetTrunk(context.Background(), cpu, fixture.Frames)
	if err != nil {
		t.Fatal(err)
	}
	gotFeatures, gotShape, err := trunk.ForwardFrames(context.Background(), fixture.Input, fixture.Frames)
	if err != nil || gotShape != wantShape {
		nativeCommunityClose(t, trunk)
		t.Fatal(gotShape, wantShape, err)
	}
	nativeCommunityCompare(t, "resnet-trunk", gotFeatures, wantFeatures, 5e-4, 5e-4)
	nativeCommunityClose(t, trunk)

	hybrid, err := NewVulkanEmbedding(context.Background(), cpu, fixture.Frames)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeCommunityClose(t, hybrid)
	for i, mask := range fixture.MaskCases {
		want, err := cpu.Forward(context.Background(), fixture.Input, fixture.Frames, mask.Masks, mask.Speakers, mask.MaskFrames, WeSpeakerBlockGEMM)
		if err != nil {
			t.Fatal(err)
		}
		got, err := hybrid.Forward(context.Background(), fixture.Input, fixture.Frames, mask.Masks, mask.Speakers, mask.MaskFrames)
		if err != nil {
			t.Fatal(err)
		}
		nativeCommunityCompare(t, "hybrid-embedding", got.Embeddings, want.Embeddings, 1e-3, 1e-3)
		if !reflectIntSlice(got.NonzeroFrames, want.NonzeroFrames) {
			t.Fatal("hybrid support", i)
		}
	}
}

func reflectIntSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func nativeCommunityLSTM(t *testing.T) {
	fixture := loadLSTMFixtures(t)[2]
	cpu, err := NewLSTM(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	want, err := cpu.Forward(context.Background(), fixture.Input, fixture.Frames, fixture.InitialHidden, fixture.InitialCell, LSTMSIMD)
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := NewVulkanLSTM(context.Background(), cpu, fixture.Frames)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeCommunityClose(t, gpu)
	got, err := gpu.Forward(context.Background(), fixture.Input, fixture.InitialHidden, fixture.InitialCell)
	if err != nil {
		t.Fatal(err)
	}
	nativeCommunityCompare(t, "lstm-output", got.Output, want.Output, 2e-4, 2e-4)
	nativeCommunityCompare(t, "lstm-hidden", got.Hidden, want.Hidden, 2e-4, 2e-4)
	nativeCommunityCompare(t, "lstm-cell", got.Cell, want.Cell, 2e-4, 2e-4)
}

func nativeCommunitySegmentationFeatures(t *testing.T) {
	fixture := segmentationLoadOracles(t)[0]
	checkpoint, err := LoadSegmentationSource(context.Background(), segmentationFixtureSource(fixture), fixture.Config)
	if err != nil {
		t.Fatal(err)
	}
	want, err := checkpoint.ForwardFeatures(context.Background(), fixture.Features, fixture.Frames, LSTMSIMD, HeadSIMD)
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := NewVulkanSegmentationFeatures(context.Background(), checkpoint, fixture.Frames)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeCommunityClose(t, gpu)
	got, err := gpu.ForwardFeatures(context.Background(), fixture.Features, fixture.Frames, HeadSIMD)
	if err != nil {
		t.Fatal(err)
	}
	nativeCommunityCompare(t, "segmentation-features", got, want, 5e-4, 5e-4)
}

func nativeCommunitySegmentationPCM(t *testing.T) {
	cpu, pcm := experimentalFixture(t)
	filters := append([]float32(nil), cpu.frontend.filters...)
	want, err := cpu.ForwardPCM(context.Background(), pcm, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD})
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := NewVulkanSegmentationPCM(context.Background(), cpu.checkpoint, filters, len(pcm))
	if err != nil {
		t.Fatal(err)
	}
	defer nativeCommunityClose(t, gpu)
	got, err := gpu.ForwardPCM(context.Background(), pcm, SincNetSIMDFMA, HeadSIMD)
	if err != nil || got.Grid != want.Grid || got.Classes != want.Classes {
		t.Fatal(got, err)
	}
	nativeCommunityCompare(t, "segmentation-pcm", got.LogProbabilities, want.LogProbabilities, 5e-4, 5e-4)
}
