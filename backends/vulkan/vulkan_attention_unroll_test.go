package vulkan

import (
	"context"
	"errors"
	"testing"
)

func TestVulkanOfflineAttentionUnrollContract(t *testing.T) {
	c, e := InspectVulkanShader(spirv_attention_f32_key32_scoreilp_unroll4)
	if e != nil {
		t.Fatal(e)
	}
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 14528, StorageBindings: 15, PushBytes: 20}
	if c != want {
		t.Fatal(c, want)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if o, e := NewVkAttentionKey32ScoreILPUnroll4F32(ctx); o != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("cancel", e)
	}
	for _, dim := range []int{1, 32, 63, 65} {
		if vkAttentionHeadDim(64, dim) == nil {
			t.Fatal("dimension", dim)
		}
	}
	if vkAttentionHeadDim(64, 64) != nil || vkAttentionHeadDim(0, 32) != nil {
		t.Fatal("admission")
	}
	offlineVK(t)
	mockVK(t, &vkLimits, offlineLimits())
	vkLimits.SharedMemoryBytes = 14527
	if _, e := NewVkAttentionKey32ScoreILPUnroll4F32(context.Background()); !errors.Is(e, ErrVulkanLimit) {
		t.Fatal("prealloc", e)
	}
}
