package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingReLU(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingReLU)
	if err != nil || c.StorageBindings != 7 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 8
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{relu: k}
	x, seed, dst := linearMetadataTensor(1, 257), linearMetadataTensor(2, 257), linearMetadataTensor(3, 257)
	for mode := uint32(0); mode < 2; mode++ {
		s, err := op.ReLUStage(context.Background(), dst, x, seed, mode)
		if err != nil || s.Groups != [3]uint32{2, 1, 1} || s.PushWords[1] != mode {
			t.Fatal(s, err)
		}
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.ReLUStage(context.Background(), dst, x, seed, 0); err == nil {
			t.Fatal("bad owner")
		}
	}
	for _, b := range [][3]*VkTensorF32{{x, x, seed}, {dst, nil, seed}, {dst, x, nil}, {nil, x, seed}, {dst, x, linearMetadataTensor(4, 256)}, {linearMetadataTensor(4, 1, 257), x, seed}} {
		if _, err := op.ReLUStage(context.Background(), b[0], b[1], b[2], 0); err == nil {
			t.Fatal("bad extent/alias")
		}
	}
	if _, err := op.ReLUStage(context.Background(), dst, x, seed, 2); err == nil {
		t.Fatal("bad mode")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.ReLUStage(ctx, dst, x, seed, 0); err == nil {
		t.Fatal("cancel")
	}
	if _, err := op.tapeStage(context.Background(), []*VkTensorF32{x, seed, dst}, train.Command{Op: "relu", Inputs: []int{0, 1}, Output: 2}); err != nil {
		t.Fatal(err)
	}
	if op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "relu"}}}) != nil {
		t.Fatal("preflight")
	}
}
