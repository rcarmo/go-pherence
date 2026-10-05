package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingEmbedding(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingEmbedding)
	if err != nil || c.StorageBindings != 7 || c.PushBytes != 16 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{embedding: k}
	ids := linearMetadataTensor(2, 3)
	for mode := uint32(0); mode < 4; mode++ {
		ids = linearMetadataTensor(2, 3)
		if mode == 2 {
			ids = linearMetadataTensor(2, 8)
		}
		if mode == 3 {
			ids = linearMetadataTensor(2, 10)
		}
		a, n, _ := train.EmbeddingShape(3, 2, 4, mode)
		source, dst := linearMetadataTensor(1, a), linearMetadataTensor(3, n)
		s, err := op.EmbeddingStage(context.Background(), dst, source, ids, 3, 2, 4, mode)
		if err != nil || s.PushWords[3] != mode || s.Groups != [3]uint32{1, 1, 1} {
			t.Fatal(s, err)
		}
		if _, err := op.tapeStage(context.Background(), []*VkTensorF32{source, ids, dst}, train.Command{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 4, Mode: mode}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := op.EmbeddingStage(context.Background(), linearMetadataTensor(3, 8), linearMetadataTensor(1, 6), linearMetadataTensor(2, 3), 3, 2, 4, 2); err == nil {
		t.Fatal("short index")
	}
	if _, err := op.EmbeddingStage(context.Background(), linearMetadataTensor(3, 8), linearMetadataTensor(1, 6), linearMetadataTensor(2, 8), 0, 2, 4, 2); err == nil {
		t.Fatal("index shape admission")
	}
	if _, err := op.EmbeddingStage(context.Background(), linearMetadataTensor(3, 6), linearMetadataTensor(1, 6), linearMetadataTensor(2, 8), 3, 2, 4, 3); err == nil {
		t.Fatal("short compact index")
	}
	ids = linearMetadataTensor(2, 3)
	source, dst := linearMetadataTensor(1, 8), linearMetadataTensor(3, 6)
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.EmbeddingStage(context.Background(), dst, source, ids, 3, 2, 4, 0); err == nil {
			t.Fatal("owner")
		}
	}
	for _, b := range [][3]*VkTensorF32{{source, source, ids}, {dst, nil, ids}, {nil, source, ids}, {dst, source, nil}, {dst, source, linearMetadataTensor(4, 2)}, {linearMetadataTensor(4, 2, 3), source, ids}} {
		if _, err := op.EmbeddingStage(context.Background(), b[0], b[1], b[2], 3, 2, 4, 0); err == nil {
			t.Fatal("bad extent/alias")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.EmbeddingStage(ctx, dst, source, ids, 3, 2, 4, 0); err == nil {
		t.Fatal("cancel")
	}
	if _, err := op.EmbeddingStage(ctx, linearMetadataTensor(3, 6), linearMetadataTensor(1, 6), linearMetadataTensor(2, 10), 3, 2, 4, 3); err == nil {
		t.Fatal("compact cancellation")
	}
	if op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "embedding"}}}) != nil {
		t.Fatal("preflight")
	}
}
