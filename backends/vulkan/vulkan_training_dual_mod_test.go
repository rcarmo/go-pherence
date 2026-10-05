package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDualMod(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingModDualBackward)
	if err != nil || c.StorageBindings != 127 || c.PushBytes != 12 {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 7
	kernel.pushSize = 12
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{modDual: kernel}
	for kind := uint32(0); kind < 2; kind++ {
		for mode := uint32(0); mode < 4; mode++ {
			n, bn, count := 3, 3, 3
			if kind == 0 {
				bn = 6
				if mode >= 2 {
					count = 6
				}
			}
			a, at, b, bt, gv, gt, dst := linearMetadataTensor(1, n), linearMetadataTensor(2, n), linearMetadataTensor(3, bn), linearMetadataTensor(4, bn), linearMetadataTensor(5, n), linearMetadataTensor(6, n), linearMetadataTensor(7, count)
			stage, err := op.ModDualBackwardStage(context.Background(), dst, a, at, b, bt, gv, gt, kind, mode)
			if err != nil || stage.PushWords[1] != kind || stage.PushWords[2] != mode {
				t.Fatal(stage, err)
			}
			if mode < 2 {
				if _, err := op.ModDualBackwardStage(context.Background(), a, a, at, b, bt, gv, gt, kind, mode); err == nil {
					t.Fatal("mod alias")
				}
			}
		}
	}
}
