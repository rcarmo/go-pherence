package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingTransformerPrimitives(t *testing.T) {
	for _, spec := range []struct {
		code []byte
		push uint32
	}{{spirvTrainingGELU, 8}, {spirvTrainingRow, 20}} {
		c, err := InspectVulkanShader(spec.code)
		if err != nil || c.PushBytes != spec.push || c.StorageBindings != 7 || c.SharedBytes != 0 || c.LocalSize != [3]uint32{256, 1, 1} {
			t.Fatal(c, err)
		}
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 3
	kernel.pushSize = 8
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{gelu: kernel, row: kernel}
	x, seed, dst := linearMetadataTensor(1, 257), linearMetadataTensor(2, 257), linearMetadataTensor(3, 257)
	for mode := uint32(0); mode < 2; mode++ {
		s, err := op.GELUStage(context.Background(), dst, x, seed, mode)
		if err != nil || s.Groups != [3]uint32{2, 1, 1} || s.PushWords[0] != 257 || s.PushWords[1] != mode {
			t.Fatal(s, err)
		}
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.GELUStage(context.Background(), dst, x, seed, 0); err == nil {
			t.Fatal("nil owner")
		}
	}
	for _, inputs := range [][3]*VkTensorF32{{x, x, seed}, {dst, nil, seed}, {dst, x, nil}, {nil, x, seed}, {dst, x, linearMetadataTensor(4, 256)}, {linearMetadataTensor(4, 1, 257), x, seed}} {
		if _, err := op.GELUStage(context.Background(), inputs[0], inputs[1], inputs[2], 0); err == nil {
			t.Fatal("GELU extent")
		}
	}
	if _, err := op.GELUStage(context.Background(), dst, x, seed, 2); err == nil {
		t.Fatal("mode")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.GELUStage(ctx, dst, x, seed, 0); err == nil {
		t.Fatal("cancel")
	}
	if _, err := op.tapeStage(context.Background(), []*VkTensorF32{x, seed, dst}, train.Command{Op: "gelu", Inputs: []int{0, 1}, Output: 2}); err != nil {
		t.Fatal(err)
	}
	kernel.pushSize = 20
	for mode := uint32(0); mode < 4; mode++ {
		parts, slot := 1, 0
		if mode < 2 {
			parts, slot = 3, 2
		}
		an, bn, n, _ := train.RowShape(3, 5, parts, slot, mode)
		a, b, o := linearMetadataTensor(1, an), linearMetadataTensor(2, bn), linearMetadataTensor(3, n)
		s, err := op.RowStage(context.Background(), o, a, b, 3, 5, parts, slot, mode)
		if err != nil || s.PushWords[2] != uint32(parts) || s.PushWords[3] != uint32(slot) || s.PushWords[4] != mode {
			t.Fatal(s, err)
		}
		if _, err := op.RowStage(context.Background(), a, a, b, 3, 5, parts, slot, mode); err == nil {
			t.Fatal("row alias")
		}
		if _, err := op.RowStage(context.Background(), linearMetadataTensor(4, n+1), a, b, 3, 5, parts, slot, mode); err == nil {
			t.Fatal("row extent")
		}
		if _, err := op.tapeStage(context.Background(), []*VkTensorF32{a, b, o}, train.Command{Op: "row", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 5, Parts: parts, Slot: slot, Mode: mode}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := op.RowStage(context.Background(), dst, x, seed, 0, 5, 1, 0, 2); err == nil {
		t.Fatal("row dimensions")
	}
	if _, err := op.RowStage(ctx, dst, x, seed, 3, 5, 1, 0, 2); err == nil {
		t.Fatal("cancel")
	}
	var nilOp *VkTrainingF32
	if _, err := nilOp.RowStage(context.Background(), dst, x, seed, 3, 5, 1, 0, 2); err == nil {
		t.Fatal("nil row owner")
	}
	if err := op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "gelu"}, {Op: "row"}}}); err != nil {
		t.Fatal(err)
	}
}
