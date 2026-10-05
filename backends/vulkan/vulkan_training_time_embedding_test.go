package vulkan

import (
	"context"
	"encoding/binary"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingTimeEmbedding(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingTimeEmbedding)
	if err != nil || c.StorageBindings != 31 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 5
	k.pushSize = 8
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{timeEmbedding: k}
	f, time, gv, gt := linearMetadataTensor(1, 257), linearMetadataTensor(2, 2), linearMetadataTensor(3, 514), linearMetadataTensor(4, 514)
	for mode := uint32(0); mode < 3; mode++ {
		n, _ := train.TimeEmbeddingShape(257, mode)
		dst := linearMetadataTensor(5, n)
		s, err := op.TimeEmbeddingStage(context.Background(), dst, f, time, gv, gt, mode)
		groups := uint32(3)
		if mode == 2 {
			groups = 1
		}
		if err != nil || s.PushWords[1] != mode || s.Groups != [3]uint32{groups, 1, 1} {
			t.Fatal(s, err)
		}
		if _, err := op.tapeStage(context.Background(), []*VkTensorF32{f, time, gv, gt, dst}, train.Command{Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 4, Mode: mode}); err != nil {
			t.Fatal(err)
		}
	}
	dst := linearMetadataTensor(5, 514)
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.TimeEmbeddingStage(context.Background(), dst, f, time, gv, gt, 0); err == nil {
			t.Fatal("owner")
		}
	}
	for _, bad := range []struct{ f, time, gv, gt, dst *VkTensorF32 }{{nil, time, gv, gt, dst}, {f, nil, gv, gt, dst}, {f, time, nil, gt, dst}, {f, time, gv, nil, dst}, {f, time, gv, gt, nil}, {f, time, gv, gt, f}, {f, linearMetadataTensor(6, 1), gv, gt, dst}, {f, time, gv, gt, linearMetadataTensor(6, 2, 257)}} {
		if _, err := op.TimeEmbeddingStage(context.Background(), bad.dst, bad.f, bad.time, bad.gv, bad.gt, 0); err == nil {
			t.Fatal("extent/alias")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.TimeEmbeddingStage(ctx, dst, f, time, gv, gt, 0); err == nil {
		t.Fatal("cancel")
	}
	if op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "time-embedding"}}}) != nil {
		t.Fatal("preflight")
	}
	// FAbs remains a unary instruction in the replacement; Tan still rejects.
	code := append([]byte(nil), spirvTrainingTimeEmbedding...)
	changed := false
	for pos := 20; pos < len(code); {
		word := binary.LittleEndian.Uint32(code[pos:])
		n := int(word >> 16)
		if uint16(word) == 12 {
			ext := binary.LittleEndian.Uint32(code[pos+16:])
			if ext == 4 {
				binary.LittleEndian.PutUint32(code[pos+16:], 15)
				changed = true
				break
			}
		}
		pos += 4 * n
	}
	if !changed {
		t.Fatal("absolute-value instruction missing")
	}
	if _, err := InspectVulkanShader(code); err == nil {
		t.Fatal("Tan admitted")
	}
}
