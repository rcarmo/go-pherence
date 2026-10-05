package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingRoPE(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingRoPE)
	if err != nil || c.LocalSize != [3]uint32{256, 1, 1} || c.PushBytes != 16 || c.StorageBindings != 7 || c.SharedBytes != 0 {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 3
	kernel.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{rope: kernel}
	x, table, dst := linearMetadataTensor(1, 13, 3, 14), linearMetadataTensor(2, 13, 7, 2), linearMetadataTensor(3, 13, 3, 14)
	for mode := uint32(0); mode < 2; mode++ {
		s, err := op.RoPEStage(context.Background(), dst, x, table, mode)
		if err != nil || s.Groups != [3]uint32{2, 1, 1} || !reflect.DeepEqual(s.PushWords, []uint32{13, 3, 14, mode}) {
			t.Fatal(s, err)
		}
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.RoPEStage(context.Background(), dst, x, table, 0); err == nil {
			t.Fatal("unowned kernel")
		}
	}
	for _, inputs := range [][3]*VkTensorF32{{x, x, table}, {table, x, table}, {dst, nil, table}, {dst, x, nil}, {nil, x, table}, {linearMetadataTensor(4, 13, 42), x, table}, {dst, x, linearMetadataTensor(4, 13, 6, 2)}, {linearMetadataTensor(4, 13, 2, 14), x, table}} {
		if _, err := op.RoPEStage(context.Background(), inputs[0], inputs[1], inputs[2], 0); err == nil {
			t.Fatal("alias/rank/shape")
		}
	}
	if _, err := op.RoPEStage(context.Background(), dst, x, table, 2); err == nil {
		t.Fatal("mode")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.RoPEStage(ctx, dst, x, table, 0); err == nil {
		t.Fatal("cancelled")
	}
	tensors := []*VkTensorF32{x, table, dst}
	if _, err := op.tapeStage(context.Background(), tensors, train.Command{Op: "rope", Inputs: []int{0, 1}, Output: 2, Mode: 1}); err != nil {
		t.Fatal("tape routing", err)
	}
}
func TestVulkanOfflineTrainingRoPEOwner(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	calls := 0
	op, err := newVkTrainingF32Locked(func(code []byte, buffers, push int) (*VkComputeKernel, error) {
		calls++
		if calls == 16 && (!reflect.DeepEqual(code, spirvTrainingRoPE) || buffers != 3 || push != 16) {
			t.Fatal("RoPE creation contract")
		}
		if calls == 27 && (!reflect.DeepEqual(code, spirvTrainingAdamWScaled) || buffers != 6 || push != 40) {
			t.Fatal("device-scale AdamW creation contract")
		}
		if calls == 28 && (!reflect.DeepEqual(code, spirvTrainingEmbeddingAccumulate) || buffers != 4 || push != 16) {
			t.Fatal("compact accumulation creation contract")
		}
		if calls == 29 && (!reflect.DeepEqual(code, spirvTrainingEOSAdmission) || buffers != 4 || push != 12) {
			t.Fatal("EOS admission creation contract")
		}
		return kernel, nil
	})
	if err != nil || calls != 29 || op.rope != kernel || op.adamScaled != kernel || op.embeddingAccumulate != kernel {
		t.Fatal(op, err, calls)
	}
	if err := op.Close(); err != nil || !kernel.closed {
		t.Fatal("close lost RoPE", err)
	}
}
