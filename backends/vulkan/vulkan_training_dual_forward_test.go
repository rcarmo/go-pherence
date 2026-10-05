package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDualNormForward(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingNormDualForward)
	if err != nil || c.StorageBindings != 63 || c.PushBytes != 20 {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 6
	kernel.pushSize = 20
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{normDualForward: kernel}
	x, xt, w, bias, stats := linearMetadataTensor(1, 3, 7), linearMetadataTensor(2, 3, 7), linearMetadataTensor(3, 7), linearMetadataTensor(4, 7), linearMetadataTensor(5, 3, 5)
	for kind := uint32(0); kind < 2; kind++ {
		for mode := uint32(0); mode < 3; mode++ {
			dst := linearMetadataTensor(6, 3, 7)
			if mode == 0 {
				dst = linearMetadataTensor(6, 3, 5)
			}
			s, err := op.NormDualForwardStage(context.Background(), dst, x, xt, w, bias, stats, kind, mode, 1e-4)
			if err != nil || s.PushWords[2] != kind || s.PushWords[3] != mode {
				t.Fatal(s, err)
			}
		}
	}
	if _, err := op.NormDualForwardStage(context.Background(), stats, x, xt, w, bias, stats, 0, 0, 1e-4); err == nil {
		t.Fatal("stats read/write overlap")
	}
}
func TestVulkanOfflineTrainingDualNormForwardPlan(t *testing.T) {
	_, kernel, _, _ := newPlanMock(t)
	kernel.numBuffers = 6
	kernel.pushSize = 20
	op := &VkTrainingF32{normDualForward: kernel}
	y, yt, stats, scratch, x, xt, w, bias := linearMetadataTensor(1, 3, 7), linearMetadataTensor(2, 3, 7), linearMetadataTensor(3, 3, 5), linearMetadataTensor(4, 3, 5), linearMetadataTensor(5, 3, 7), linearMetadataTensor(6, 3, 7), linearMetadataTensor(7, 7), linearMetadataTensor(8, 7)
	plan, err := op.NormDualForwardPlan(context.Background(), y, yt, stats, scratch, x, xt, w, bias, 0, 1e-4)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.state.stages) != 3 {
		t.Fatal("forward stage count")
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
}
