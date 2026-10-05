package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingSiLUDual(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingSiLUDualBackward)
	if err != nil || c.StorageBindings != 63 || c.PushBytes != 4 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 6
	kernel.pushSize = 4
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{siluDual: kernel}
	tensors := make([]*VkTensorF32, 6)
	for i := range tensors {
		tensors[i] = linearMetadataTensor(i+1, 257)
	}
	s, err := op.SiLUDualBackwardStage(context.Background(), tensors[4], tensors[5], tensors[0], tensors[1], tensors[2], tensors[3])
	if err != nil || s.Groups[0] != 2 {
		t.Fatal(s, err)
	}
	if _, err := op.SiLUDualBackwardStage(context.Background(), tensors[4], tensors[4], tensors[0], tensors[1], tensors[2], tensors[3]); err == nil {
		t.Fatal("adjoints overlap")
	}
}
