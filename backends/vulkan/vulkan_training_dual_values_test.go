package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDualValueShaders(t *testing.T) {
	for _, tc := range []struct {
		code []byte
		b, p uint32
	}{{spirvTrainingLinearDualForward, 31, 16}, {spirvTrainingSiLUDualForward, 7, 8}, {spirvTrainingModDualForward, 31, 12}} {
		c, err := InspectVulkanShader(tc.code)
		if err != nil || c.StorageBindings != tc.b || c.PushBytes != tc.p {
			t.Fatal(c, err)
		}
	}
}
func TestVulkanOfflineTrainingDualValueBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{linearDualForward: kernel, siluDualForward: kernel, modDualForward: kernel}
	for mode := uint32(0); mode < 2; mode++ {
		kernel.numBuffers = 5
		kernel.pushSize = 16
		x, xt, w, bias, dst := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 3, 5), linearMetadataTensor(3, 7, 5), linearMetadataTensor(4, 7), linearMetadataTensor(5, 3, 7)
		if _, err := op.LinearDualForwardStage(context.Background(), dst, x, xt, w, bias, mode); err != nil {
			t.Fatal(err)
		}
		kernel.numBuffers = 3
		kernel.pushSize = 8
		x, xt, dst = linearMetadataTensor(1, 257), linearMetadataTensor(2, 257), linearMetadataTensor(5, 257)
		if _, err := op.SiLUDualForwardStage(context.Background(), dst, x, xt, mode); err != nil {
			t.Fatal(err)
		}
		if _, err := op.SiLUDualForwardStage(context.Background(), x, x, xt, mode); err == nil {
			t.Fatal("alias")
		}
		kernel.numBuffers = 5
		kernel.pushSize = 12
		for kind := uint32(0); kind < 2; kind++ {
			bn := 257
			if kind == 0 {
				bn *= 2
			}
			b, bt := linearMetadataTensor(3, bn), linearMetadataTensor(4, bn)
			if _, err := op.ModDualForwardStage(context.Background(), dst, x, xt, b, bt, kind, mode); err != nil {
				t.Fatal(err)
			}
		}
	}
}
