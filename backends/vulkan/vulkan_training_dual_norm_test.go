package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDualNormLinearShaders(t *testing.T) {
	for _, tc := range []struct {
		code     []byte
		bindings uint32
	}{{spirvTrainingLinearDualBackward, 63}, {spirvTrainingNormDualBackward, 127}} {
		c, err := InspectVulkanShader(tc.code)
		if err != nil || c.LocalSize != [3]uint32{256, 1, 1} || c.StorageBindings != tc.bindings || c.PushBytes != 16 {
			t.Fatal(c, err)
		}
	}
}
func TestVulkanOfflineTrainingDualNormBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 7
	kernel.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{normDual: kernel}
	x, xt, w, gv, gt, stats := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 3, 5), linearMetadataTensor(3, 5), linearMetadataTensor(4, 3, 5), linearMetadataTensor(5, 3, 5), linearMetadataTensor(6, 3, 5)
	for kind := uint32(0); kind < 2; kind++ {
		for mode := uint32(0); mode < 4; mode++ {
			dst := linearMetadataTensor(7, 5)
			if mode < 2 {
				dst = linearMetadataTensor(7, 3, 5)
			}
			stage, err := op.NormDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, stats, kind, mode)
			if kind == 1 && mode == 3 {
				if err == nil {
					t.Fatal("variance bias admitted")
				}
				continue
			}
			if err != nil || stage.PushWords[2] != kind || stage.PushWords[3] != mode {
				t.Fatal(stage, err)
			}
		}
	}
	if _, err := op.NormDualBackwardStage(context.Background(), x, x, xt, w, gv, gt, stats, 0, 0); err == nil {
		t.Fatal("alias admitted")
	}
	stats.shape[1] = 4
	if _, err := op.NormDualBackwardStage(context.Background(), linearMetadataTensor(7, 3, 5), x, xt, w, gv, gt, stats, 0, 0); err == nil {
		t.Fatal("stats mismatch")
	}
}
func TestVulkanOfflineTrainingDualLinearBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 6
	kernel.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{linearDual: kernel}
	x, xt, w, gv, gt := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 3, 5), linearMetadataTensor(3, 7, 5), linearMetadataTensor(4, 3, 7), linearMetadataTensor(5, 3, 7)
	for mode := uint32(0); mode < 4; mode++ {
		dst := linearMetadataTensor(6, 7)
		if mode < 2 {
			dst = linearMetadataTensor(6, 3, 5)
		} else if mode == 2 {
			dst = linearMetadataTensor(6, 7, 5)
		}
		stage, err := op.LinearDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, mode)
		if err != nil || stage.PushWords[3] != mode {
			t.Fatal(stage, err)
		}
	}
	if _, err := op.LinearDualBackwardStage(context.Background(), x, x, xt, w, gv, gt, 0); err == nil {
		t.Fatal("alias admitted")
	}
}
