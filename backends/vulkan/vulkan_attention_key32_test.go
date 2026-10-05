package vulkan

import (
	"context"
	"testing"
)

func TestVulkanOfflineAttentionKey32Contract(t *testing.T) {
	offlineVK(t)
	got, err := InspectVulkanShader(spirv_attention_f32_key32)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 14528, StorageBindings: 15, PushBytes: 20}
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewVkAttentionKey32F32(ctx); err != context.Canceled {
		t.Fatal(err)
	}
	vkLimits.SharedMemoryBytes = 14527
	_, err = NewVkAttentionKey32F32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}
