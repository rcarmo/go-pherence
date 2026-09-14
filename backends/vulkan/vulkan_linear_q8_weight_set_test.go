package vulkan

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

func TestVulkanOfflineLinearQ8WeightSetLayoutAndStages(t *testing.T) {
	_, kernel, arena, _ := newPlanMock(t)
	kernel.numBuffers, kernel.pushSize = 5, 12
	storage := arena.state.buffer
	storage.size = 256 // descriptor-only fixture; mapped bytes are never accessed
	vkLimits.StorageBufferOffsetAlignment = 64
	matrices := []VkLinearQ8WeightMatrix{
		{Weights: []float32{1, -2, 3, 4, -5, 6}, OutDim: 2, InDim: 3},
		{Weights: nativeData(5*7, 77, .5), OutDim: 5, InDim: 7},
	}
	views, packed, scales, bytes, err := prepareLinearQ8WeightSet(context.Background(), matrices, 64)
	if err != nil || len(packed) != 2 || len(scales) != 2 {
		t.Fatal(err, len(packed), len(scales))
	}
	set := &VkLinearQ8WeightSet{kernel: kernel, storage: storage, storageBytes: bytes, views: views}
	if set.Count() != 2 || set.StorageBytes() != 212 || len(set.views) != 2 {
		t.Fatal("set layout", set.Count(), set.StorageBytes(), set.views)
	}
	if a, b := set.views[0], set.views[1]; a.packedOffset != 0 || a.packedBytes != 8 || a.scaleOffsetBytes != 64 || b.packedOffset != 128 || b.packedBytes != 36 || b.scaleOffsetBytes != 192 {
		t.Fatal("aligned views", a, b)
	}
	x0, b0, o0 := linearMetadataTensor(11, 4, 3), linearMetadataTensor(12, 2), linearMetadataTensor(13, 4, 2)
	x1, b1, o1 := linearMetadataTensor(14, 4, 7), linearMetadataTensor(15, 5), linearMetadataTensor(16, 4, 5)
	s0, err := set.Stage(context.Background(), 0, o0, x0, b0)
	if err != nil {
		t.Fatal(err)
	}
	s1, err := set.Stage(context.Background(), 1, o1, x1, b1)
	if err != nil {
		t.Fatal(err)
	}
	if s0.Kernel != s1.Kernel || s0.bindings[1].buffer != s1.bindings[1].buffer || s0.bindings[1].offset != 0 || s0.bindings[2].offset != 64 || s1.bindings[1].offset != 128 || s1.bindings[2].offset != 192 {
		t.Fatal("shared Q8 owner stages")
	}
	if _, err := set.Stage(context.Background(), -1, o0, x0, b0); err == nil {
		t.Fatal("negative index")
	}
	if _, err := set.Stage(context.Background(), 2, o0, x0, b0); err == nil {
		t.Fatal("high index")
	}
	if _, err := set.Stage(context.Background(), 0, o1, x1, b1); err == nil {
		t.Fatal("wrong view shapes")
	}
	copySet := *set
	if err := set.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := copySet.Stage(context.Background(), 0, o0, x0, b0); !errors.Is(err, ErrVulkanClosed) {
		t.Fatal("copied set remained usable", err)
	}
	if err := copySet.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineLinearQ8WeightSetAdmission(t *testing.T) {
	for _, matrices := range [][]VkLinearQ8WeightMatrix{
		nil,
		{{Weights: []float32{1}, OutDim: 0, InDim: 1}},
		{{Weights: []float32{1}, OutDim: 1, InDim: 2}},
		{{Weights: []float32{float32(math.NaN())}, OutDim: 1, InDim: 1}},
		{{Weights: []float32{float32(math.Inf(1))}, OutDim: 1, InDim: 1}},
	} {
		if views, packed, scales, bytes, err := prepareLinearQ8WeightSet(context.Background(), matrices, 64); err == nil || views != nil || packed != nil || scales != nil || bytes != 0 {
			t.Fatal("invalid set admitted", matrices, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if views, _, _, _, err := prepareLinearQ8WeightSet(ctx, []VkLinearQ8WeightMatrix{{Weights: []float32{1}, OutDim: 1, InDim: 1}}, 64); !errors.Is(err, context.Canceled) || views != nil {
		t.Fatal("precancel", views, err)
	}
	if err := (*VkLinearQ8WeightSet)(nil).Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineLinearQ8WeightSetPlanRetention(t *testing.T) {
	lane, kernel, _, _ := newPlanMock(t)
	kernel.numBuffers, kernel.pushSize = 5, 12
	storage, err := VkBufAlloc(32)
	if err != nil {
		t.Fatal(err)
	}
	set := &VkLinearQ8WeightSet{kernel: kernel, storage: storage, storageBytes: 24, views: []vkLinearQ8WeightView{{inDim: 2, outDim: 2, weightValues: 4, packedBytes: 4, scaleOffsetBytes: 16, scaleBytes: 8}}}
	a := mustArena(t, 64)
	x, bias, out := mustTensor(t, a, 2, 2), mustTensor(t, a, 2), mustTensor(t, a, 2, 2)
	stage, err := set.Stage(context.Background(), 0, out, x, bias)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewVkF32Plan(context.Background(), []VkF32Stage{stage})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	lane.hook = func(event string) {
		if event == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	expectErrorIs(t, set.Close(), ErrVulkanInFlight)
	lane.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if err := set.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}
