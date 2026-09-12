package community1

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestVulkanSegmentationPCMRejectsBeforeDevice(t *testing.T) {
	model, pcm := experimentalFixture(t)
	checkpoint := model.checkpoint
	filters := append([]float32(nil), model.frontend.filters...)
	called := false
	stop := errors.New("stop before Vulkan")
	factory := func(_ context.Context, got *SegmentationCheckpoint, frames int) (*VulkanSegmentationFeatures, error) {
		called = true
		if got != checkpoint || frames <= 0 {
			t.Fatal("PCM constructor passed wrong source/frames")
		}
		return nil, stop
	}
	result, err := newVulkanSegmentationPCM(context.Background(), checkpoint, filters, len(pcm), factory)
	if !errors.Is(err, stop) || result != nil || !called {
		t.Fatal("valid PCM composition did not reach feature constructor", result, err)
	}
	called = false
	if result, err := newVulkanSegmentationPCM(nil, checkpoint, filters, len(pcm), factory); err == nil || result != nil || called {
		t.Fatal("nil context")
	}
	if result, err := newVulkanSegmentationPCM(context.Background(), nil, filters, len(pcm), factory); err == nil || result != nil || called {
		t.Fatal("nil checkpoint")
	}
	if result, err := newVulkanSegmentationPCM(context.Background(), checkpoint, nil, len(pcm), factory); err == nil || result != nil || called {
		t.Fatal("nil filters")
	}
	if result, err := newVulkanSegmentationPCM(context.Background(), checkpoint, filters, 0, factory); err == nil || result != nil || called {
		t.Fatal("bad sample count")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := newVulkanSegmentationPCM(ctx, checkpoint, filters, len(pcm), factory); !errors.Is(err, context.Canceled) || result != nil || called {
		t.Fatal("cancelled constructor", err)
	}
	if result, err := newVulkanSegmentationPCM(context.Background(), checkpoint, filters, len(pcm), nil); err == nil || result != nil {
		t.Fatal("nil feature constructor")
	}
	if result, err := newVulkanSegmentationPCM(context.Background(), checkpoint, filters, len(pcm), func(context.Context, *SegmentationCheckpoint, int) (*VulkanSegmentationFeatures, error) {
		return nil, nil
	}); err == nil || result != nil {
		t.Fatal("nil successful feature owner")
	}
}

func TestVulkanSegmentationPCMAdmissionAndClose(t *testing.T) {
	frontend := &SincNet{}
	featureState := &vulkanSegmentationState{gate: make(chan struct{}, 1), stopping: true}
	m := &VulkanSegmentationPCM{frontend: frontend, features: &VulkanSegmentationFeatures{s: featureState}, samples: 4, grid: SincNetGrid{Frames: 1}}
	for _, test := range []struct {
		pcm      []float32
		sincMode SincNetMode
		headMode HeadMode
	}{{nil, SincNetSIMDFMA, HeadSIMD}, {make([]float32, 4), SincNetSIMD, HeadSIMD}, {make([]float32, 4), SincNetSIMDFMA, HeadMode(9)}, {[]float32{float32(math.NaN()), 0, 0, 0}, SincNetSIMDFMA, HeadSIMD}} {
		if result, err := m.ForwardPCM(context.Background(), test.pcm, test.sincMode, test.headMode); err == nil || result != nil {
			t.Fatal("invalid PCM hybrid admission", test)
		}
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if m.frontend != nil || m.features != nil {
		t.Fatal("PCM hybrid close")
	}
	if result, err := m.ForwardPCM(context.Background(), make([]float32, 4), SincNetSIMDFMA, HeadSIMD); err == nil || result != nil {
		t.Fatal("forward after close")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	var zero *VulkanSegmentationPCM
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if result, err := zero.ForwardPCM(context.Background(), nil, SincNetSIMDFMA, HeadSIMD); err == nil || result != nil {
		t.Fatal("zero PCM hybrid")
	}
	if !featureState.closed {
		t.Fatal("nested feature owner not closed")
	}
}
