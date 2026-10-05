package vulkan

import (
	"context"
	"math"
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingBackwardShaders(t *testing.T) {
	for _, tc := range []struct {
		code           []byte
		bindings, push uint32
	}{{spirvTrainingLayerNormBackward, 31, 12}, {spirvTrainingAttentionBackward, 127, 24}} {
		c, err := InspectVulkanShader(tc.code)
		if err != nil || c.LocalSize != [3]uint32{256, 1, 1} || c.SharedBytes != 0 || c.StorageBindings != tc.bindings || c.PushBytes != tc.push {
			t.Fatal(c, err)
		}
	}
}
func TestVulkanOfflineTrainingNormBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 5
	kernel.pushSize = 12
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{norm: kernel}
	x, inv, weight, dy := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 3), linearMetadataTensor(3, 5), linearMetadataTensor(4, 3, 5)
	for mode := uint32(0); mode < 3; mode++ {
		dst := linearMetadataTensor(5, 5)
		if mode == 0 {
			dst = linearMetadataTensor(5, 3, 5)
		}
		stage, err := op.LayerNormBackwardStage(context.Background(), dst, x, inv, weight, dy, mode)
		if err != nil || !reflect.DeepEqual(stage.PushWords, []uint32{3, 5, mode}) {
			t.Fatal(stage, err)
		}
	}
	if _, err := op.LayerNormBackwardStage(context.Background(), x, x, inv, weight, dy, 0); err == nil {
		t.Fatal("alias admitted")
	}
	weight.shape[0]--
	if _, err := op.LayerNormBackwardStage(context.Background(), linearMetadataTensor(5, 3, 5), x, inv, weight, dy, 0); err == nil {
		t.Fatal("shape admitted")
	}
}
func TestVulkanOfflineTrainingAttentionBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 7
	kernel.pushSize = 24
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{attention: kernel}
	q, k, v, dy := linearMetadataTensor(1, 17, 2, 3), linearMetadataTensor(2, 17, 2, 3), linearMetadataTensor(3, 17, 2, 3), linearMetadataTensor(4, 17, 2, 3)
	p, aux := linearMetadataTensor(5, 2, 17, 17), linearMetadataTensor(6, 2, 17, 17)
	for mode := uint32(0); mode < 5; mode++ {
		dst := linearMetadataTensor(7, 17, 2, 3)
		count := 17 * 2 * 3
		if mode < 2 {
			dst = linearMetadataTensor(7, 2, 17, 17)
			count = 2 * 17 * 17
			if mode == 1 {
				count = 2 * 17
			}
		}
		stage, err := op.AttentionBackwardStage(context.Background(), dst, q, k, v, p, dy, aux, 4, mode, .5)
		if err != nil || stage.Groups != [3]uint32{uint32((count + 255) / 256), 1, 1} || !reflect.DeepEqual(stage.PushWords, []uint32{17, 2, 3, 4, mode, math.Float32bits(.5)}) {
			t.Fatal(mode, stage, err)
		}
	}
	// Shared read-only aux is legal when mode 0/4 does not consume it; the
	// writable destination remains distinct from every bound input.
	for _, mode := range []uint32{0, 4} {
		dst := linearMetadataTensor(7, 17, 2, 3)
		if mode == 0 {
			dst = linearMetadataTensor(7, 2, 17, 17)
		}
		if _, err := op.AttentionBackwardStage(context.Background(), dst, q, k, v, p, dy, p, 4, mode, .5); err != nil {
			t.Fatal("read-only repeated aux", mode, err)
		}
	}
	for _, args := range []struct {
		dst     *VkTensorF32
		context int
		mode    uint32
		scale   float32
	}{{q, 4, 2, .5}, {linearMetadataTensor(7, 17, 2, 3), -1, 2, .5}, {linearMetadataTensor(7, 17, 2, 3), 4, 5, .5}, {linearMetadataTensor(7, 17, 2, 3), 4, 2, 0}} {
		if _, err := op.AttentionBackwardStage(context.Background(), args.dst, q, k, v, p, dy, aux, args.context, args.mode, args.scale); err == nil {
			t.Fatal("invalid attention admitted")
		}
	}
	aux.rank = 2
	if _, err := op.AttentionBackwardStage(context.Background(), linearMetadataTensor(7, 17, 2, 3), q, k, v, p, dy, aux, 4, 2, .5); err == nil {
		t.Fatal("aux mismatch admitted")
	}
}
