package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingLSDLoss(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingLSDLoss)
	if err != nil || c.StorageBindings != 15 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 4
	k.pushSize = 8
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{lsdLoss: k}
	p, d, l := linearMetadataTensor(1, 257), linearMetadataTensor(2, 257), linearMetadataTensor(3, 1)
	for mode := uint32(0); mode < 3; mode++ {
		n, _ := train.LSDLossShape(257, mode)
		out := linearMetadataTensor(4, n)
		s, err := op.LSDLossStage(context.Background(), out, p, d, l, mode)
		groups := uint32(1)
		if mode == 1 {
			groups = 2
		}
		if err != nil || s.PushWords[1] != mode || s.Groups != [3]uint32{groups, 1, 1} {
			t.Fatal(s, err)
		}
		if _, err := op.tapeStage(context.Background(), []*VkTensorF32{p, d, l, out}, train.Command{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: mode}); err != nil {
			t.Fatal(err)
		}
	}
	out := linearMetadataTensor(4, 257)
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.LSDLossStage(context.Background(), out, p, d, l, 1); err == nil {
			t.Fatal("owner")
		}
	}
	for _, bad := range []struct{ p, d, l, out *VkTensorF32 }{{nil, d, l, out}, {p, nil, l, out}, {p, d, nil, out}, {p, d, l, nil}, {p, d, l, p}, {p, d, linearMetadataTensor(5, 2), out}, {p, d, l, linearMetadataTensor(5, 1, 257)}} {
		if _, err := op.LSDLossStage(context.Background(), bad.out, bad.p, bad.d, bad.l, 1); err == nil {
			t.Fatal("extent/alias")
		}
	}
	if _, err := op.LSDLossStage(context.Background(), out, p, d, l, 3); err == nil {
		t.Fatal("mode")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.LSDLossStage(ctx, out, p, d, l, 1); err == nil {
		t.Fatal("cancel")
	}
	if op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "lsd-loss"}}}) != nil {
		t.Fatal("preflight")
	}
}
