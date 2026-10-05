package vulkan

import (
	"context"
	"testing"
)

func TestVulkanOfflineAttentionScoreILPContract(t *testing.T) {
	got, err := InspectVulkanShader(spirv_attention_f32_key32_scoreilp)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 14528, StorageBindings: 15, PushBytes: 20}
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	if err := vkCheckShaderInterface(got, 4, 20); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	op, err := NewVkAttentionKey32ScoreILPF32(ctx)
	if op != nil || err == nil {
		t.Fatal("cancel admission", op, err)
	}
	offlineVK(t)
	vkLimits.SharedMemoryBytes = 14527
	op, err = NewVkAttentionKey32ScoreILPF32(context.Background())
	if op != nil {
		t.Fatal("limit admitted")
	}
	expectErrorIs(t, err, ErrVulkanLimit)
}
