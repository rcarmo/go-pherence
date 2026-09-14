package community1

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

func TestVulkanSegmentationRejectsBeforeDevice(t *testing.T) {
	fixture := segmentationLoadOracles(t)[0]
	checkpoint, err := LoadSegmentationSource(context.Background(), segmentationFixtureSource(fixture), fixture.Config)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	factory := func(context.Context, *LSTM, int) (*VulkanLSTM, error) {
		called = true
		return nil, errors.New("stop before Vulkan")
	}
	result, err := newVulkanSegmentationFeatures(context.Background(), checkpoint, fixture.Frames, factory)
	if err == nil || result != nil || !called {
		t.Fatal("valid segmentation did not reach injected constructor", result, err)
	}
	called = false
	for _, bad := range []*SegmentationCheckpoint{nil, {}, {cfg: checkpoint.cfg, recurrent: nil, head: checkpoint.head}, {cfg: checkpoint.cfg, recurrent: checkpoint.recurrent, head: nil}} {
		if result, err := newVulkanSegmentationFeatures(context.Background(), bad, fixture.Frames, factory); err == nil || result != nil || called {
			t.Fatal("invalid segmentation reached Vulkan", bad, err)
		}
	}
	if result, err := newVulkanSegmentationFeatures(nil, checkpoint, fixture.Frames, factory); err == nil || result != nil {
		t.Fatal("nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := newVulkanSegmentationFeatures(ctx, checkpoint, fixture.Frames, factory); !errors.Is(err, context.Canceled) || result != nil {
		t.Fatal("cancelled context", err)
	}
	bad := *checkpoint
	bad.head = &SegmentationHead{cfg: checkpoint.head.cfg, classes: checkpoint.head.classes, layers: append([]HeadLinear(nil), checkpoint.head.layers...), classifier: checkpoint.head.classifier}
	bad.head.classifier.Weight = append([]float32(nil), checkpoint.head.classifier.Weight...)
	bad.head.classifier.Weight[0] = float32(math.NaN())
	called = false
	if result, err := newVulkanSegmentationFeatures(context.Background(), &bad, fixture.Frames, factory); err == nil || result != nil || called {
		t.Fatal("nonfinite head reached Vulkan", err)
	}
	if result, err := newVulkanSegmentationFeatures(context.Background(), checkpoint, fixture.Frames, nil); err == nil || result != nil {
		t.Fatal("nil recurrent constructor")
	}
}

func TestVulkanSegmentationOwnershipAndAdmission(t *testing.T) {
	var zero *VulkanSegmentationFeatures
	if zero.Stats() != (VulkanSegmentationStats{}) {
		t.Fatal("zero segmentation stats")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if output, err := zero.ForwardFeatures(context.Background(), nil, 0, HeadSIMD); err == nil || output != nil {
		t.Fatal("zero segmentation forward")
	}
	recurrentState := &vulkanLSTMState{gate: make(chan struct{}, 1)}
	state := &vulkanSegmentationState{gate: make(chan struct{}, 1), recurrent: &VulkanLSTM{s: recurrentState}, head: &SegmentationHead{}, stats: VulkanSegmentationStats{Frames: 3, InputSize: 2}}
	owner := &VulkanSegmentationFeatures{s: state}
	copyOwner := *owner
	for _, test := range []struct {
		input  []float32
		frames int
		mode   HeadMode
	}{{nil, 2, HeadSIMD}, {make([]float32, 5), 3, HeadSIMD}, {make([]float32, 6), 3, HeadMode(9)}, {[]float32{float32(math.NaN()), 0, 0, 0, 0, 0}, 3, HeadSIMD}} {
		if output, err := owner.ForwardFeatures(context.Background(), test.input, test.frames, test.mode); err == nil || output != nil {
			t.Fatal("invalid segmentation admission", test)
		}
	}
	recurrentState.stopping = true
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !state.closed || state.recurrent != nil || state.head != nil {
		t.Fatal("segmentation close")
	}
	if output, err := copyOwner.ForwardFeatures(context.Background(), nil, 3, HeadSIMD); !errors.Is(err, vk.ErrVulkanClosed) || output != nil {
		t.Fatal("segmentation copy after close", err)
	}
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	state = &vulkanSegmentationState{gate: make(chan struct{}, 1)}
	owner = &VulkanSegmentationFeatures{s: state}
	state.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := owner.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-state.gate
}
