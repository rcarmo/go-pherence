package vulkan

import (
	"context"
	"math"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingGlobalNorm(t *testing.T) {
	contract, err := InspectVulkanShader(spirvTrainingGlobalNorm)
	if err != nil || contract.StorageBindings != 7 || contract.PushBytes != 16 || contract.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(contract, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	d := &VulkanResidentDriver{Operator: &VkTrainingF32{globalNorm: k}}
	a, b, out := &vkResidentBuffer{tensor: linearMetadataTensor(1, 4)}, &vkResidentBuffer{tensor: linearMetadataTensor(2, 6)}, &vkResidentBuffer{tensor: linearMetadataTensor(3, 4)}
	s, err := d.globalNormStage(context.Background(), out, a, b, 257, 513, 0, 1)
	if err != nil || s.PushWords[0] != 257 || s.PushWords[2] != math.Float32bits(513) || s.Groups != [3]uint32{1, 1, 1} {
		t.Fatal(s, err)
	}
	s, err = d.globalNormStage(context.Background(), out, a, b, 257, 513, 2, 1)
	if err != nil || s.PushWords[1] != 2 {
		t.Fatal("sum pass", s, err)
	}
	if d.ValidateNormMerge(context.Background(), out, a, b, 257, 513, 1) == nil || d.NormMerge(context.Background(), out, a, b, 257, 513, 3) == nil {
		t.Fatal("invalid merge mode")
	}
	out = &vkResidentBuffer{tensor: linearMetadataTensor(3, 4)}
	s, err = d.globalNormStage(context.Background(), out, a, a, 1, 770, 1, .25)
	if err != nil || s.PushWords[1] != 1 || s.PushWords[3] != math.Float32bits(.25) {
		t.Fatal(s, err)
	}
	if d.ValidateNormClip(context.Background(), out, a, 770, .25) != nil {
		t.Fatal("clip preflight")
	}
	if d.ValidateNormClip(context.Background(), a, a, 770, .25) == nil || d.ValidateNormMerge(context.Background(), out, a, nil, 257, 513, 0) == nil || d.ValidateNormClip(context.Background(), out, a, 0, 1) == nil {
		t.Fatal("bad range")
	}
	for _, bad := range []*VulkanResidentDriver{nil, {}, {Operator: &VkTrainingF32{}}} {
		if bad.ValidateNormClip(context.Background(), out, a, 770, .25) == nil {
			t.Fatal("bad owner")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.ValidateNormClip(ctx, out, a, 770, .25) == nil {
		t.Fatal("cancel")
	}
}
func TestVulkanOfflineGlobalNormDriverRetention(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 16
	d := &VulkanResidentDriver{Operator: &VkTrainingF32{globalNorm: k}}
	a, b, out := &vkResidentBuffer{tensor: linearMetadataTensor(1, 4)}, &vkResidentBuffer{tensor: linearMetadataTensor(2, 6)}, &vkResidentBuffer{tensor: linearMetadataTensor(3, 4)}
	if err := d.NormMerge(context.Background(), out, a, b, 257, 513, 0); err != nil {
		t.Fatal(err)
	}
	if err := d.NormMerge(context.Background(), out, a, b, 257, 513, 2); err != nil {
		t.Fatal(err)
	}
	out = &vkResidentBuffer{tensor: linearMetadataTensor(3, 4)}
	if err := d.NormClip(context.Background(), out, a, 770, .25); err != nil {
		t.Fatal(err)
	}
	mock.failure = "submit"
	if err := d.NormClip(context.Background(), out, a, 770, .25); err == nil || d.terminal == nil || len(d.retained) != 1 {
		t.Fatal("submit retained", err)
	}
	if d.Close() == nil {
		t.Fatal("failed driver close")
	}
}
