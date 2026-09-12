package community1

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

func TestVulkanEmbeddingRejectsBeforeDevice(t *testing.T) {
	fixture := loadResNetFixtures(t)[0]
	model, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	if result, err := NewVulkanEmbedding(nil, model, fixture.Frames); err == nil || result != nil {
		t.Fatal("nil context")
	}
	if result, err := NewVulkanEmbedding(context.Background(), nil, fixture.Frames); err == nil || result != nil {
		t.Fatal("nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := NewVulkanEmbedding(ctx, model, fixture.Frames); !errors.Is(err, context.Canceled) || result != nil {
		t.Fatal("cancelled constructor", err)
	}
	if result, err := NewVulkanEmbedding(context.Background(), model, 0); err == nil || result != nil {
		t.Fatal("invalid frames")
	}
	bad := *model
	bad.projection.Weight = append([]float32(nil), model.projection.Weight...)
	bad.projection.Weight[0] = float32(math.NaN())
	if result, err := NewVulkanEmbedding(context.Background(), &bad, fixture.Frames); err == nil || result != nil {
		t.Fatal("nonfinite projection")
	}
}

func TestVulkanEmbeddingOwnershipAndAdmission(t *testing.T) {
	var zero *VulkanEmbedding
	if zero.Stats() != (VulkanEmbeddingStats{}) {
		t.Fatal("zero embedding stats")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if result, err := zero.Forward(context.Background(), nil, 0, nil, 0, 0); err == nil || result != nil {
		t.Fatal("zero embedding forward")
	}
	trunkState := &vulkanResNetState{gate: make(chan struct{}, 1), stats: VulkanResNetStats{Frames: 8, Output: CHWShape{1, 1, 1}}}
	s := &vulkanEmbeddingState{gate: make(chan struct{}, 1), trunk: &VulkanResNetTrunk{s: trunkState}, projection: HeadLinear{Weight: []float32{1, 1}, Bias: []float32{0}}, embedDim: 1, stats: VulkanEmbeddingStats{Trunk: trunkState.stats, ProjectionBytes: 12}}
	owner := &VulkanEmbedding{s: s}
	copyOwner := *owner
	if owner.Stats().ProjectionBytes != 12 {
		t.Fatal("embedding stats")
	}
	for _, test := range []struct {
		frames, speakers, maskFrames int
		masks                        []float32
	}{
		{7, 0, 0, nil}, {8, 0, 1, nil}, {8, 1, 1, nil}, {8, 0, 0, []float32{1}}, {8, 1, 0, []float32{1}}, {8, 1, 1, []float32{2}},
	} {
		if result, err := owner.Forward(context.Background(), nil, test.frames, test.masks, test.speakers, test.maskFrames); err == nil || result != nil {
			t.Fatal("invalid embedding admission", test)
		}
	}
	trunkState.stopping = true
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || s.trunk != nil || len(s.projection.Weight) != 0 {
		t.Fatal("embedding close")
	}
	if result, err := copyOwner.Forward(context.Background(), nil, 8, nil, 0, 0); !errors.Is(err, vk.ErrVulkanClosed) || result != nil {
		t.Fatal("embedding copy after close", err)
	}
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	s = &vulkanEmbeddingState{gate: make(chan struct{}, 1)}
	owner = &VulkanEmbedding{s: s}
	s.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := owner.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-s.gate
}
