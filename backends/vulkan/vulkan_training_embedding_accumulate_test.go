package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingCompactEmbeddingAccumulate(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingEmbeddingAccumulate)
	if err != nil || c.StorageBindings != 15 || c.PushBytes != 16 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 4
	k.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{embeddingAccumulate: k}
	ctx := context.Background()
	p, compact, index, out := linearMetadataTensor(1, 8), linearMetadataTensor(2, 6), linearMetadataTensor(3, 10), linearMetadataTensor(4, 8)
	stage, err := op.CompactEmbeddingAccumulateStage(ctx, out, p, compact, index, 3, 2, 4, .25)
	if err != nil || stage.PushWords[0] != 3 || stage.PushWords[1] != 2 || stage.PushWords[2] != 4 || stage.PushWords[3] != math.Float32bits(.25) || stage.Groups != [3]uint32{1, 1, 1} {
		t.Fatal(stage, err)
	}
	d := &VulkanResidentDriver{Operator: op}
	wrap := func(x *VkTensorF32) train.ResidentBuffer { return &vkResidentBuffer{tensor: x} }
	if err := d.ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []struct {
		o, p, c, i *VkTensorF32
		rows       int
		scale      float32
	}{
		{out, p, compact, index, 0, .25}, {out, p, compact, index, 3, float32(math.NaN())}, {p, p, compact, index, 3, .25}, {out, p, nil, index, 3, .25}, {out, p, compact, linearMetadataTensor(5, 9), 3, .25}, {linearMetadataTensor(5, 2, 4), p, compact, index, 3, .25},
	} {
		if _, err := op.CompactEmbeddingAccumulateStage(ctx, bad.o, bad.p, bad.c, bad.i, bad.rows, 2, 4, bad.scale); err == nil {
			t.Fatal("bad stage")
		}
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.CompactEmbeddingAccumulateStage(ctx, out, p, compact, index, 3, 2, 4, .25); err == nil {
			t.Fatal("owner")
		}
	}
	if d.ValidateCompactEmbeddingAccumulate(nil, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25) == nil || (*VulkanResidentDriver)(nil).ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("driver/context")
	}
	if d.ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(index), 0, 2, 4, .25) == nil {
		t.Fatal("shape preflight")
	}
	if d.ValidateCompactEmbeddingAccumulate(ctx, nil, wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25) == nil || d.ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, float32(math.NaN())) == nil || d.ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(linearMetadataTensor(5, 9)), 3, 2, 4, .25) == nil {
		t.Fatal("preflight")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if d.ValidateCompactEmbeddingAccumulate(cancelled, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("cancel")
	}
	if _, err := op.CompactEmbeddingAccumulateStage(cancelled, out, p, compact, index, 3, 2, 4, .25); err == nil {
		t.Fatal("stage cancellation")
	}
	if err := d.ValidateCompactEmbeddingAccumulate(cancelled, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25); err == nil {
		t.Fatal("reshape cancellation")
	}
	d.terminal = context.Canceled
	if d.CompactEmbeddingAccumulate(ctx, wrap(out), wrap(p), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("terminal")
	}
}
func TestVulkanOfflineTrainingCompactEmbeddingTransferRetention(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 4
	k.pushSize = 16
	op := &VkTrainingF32{adam: k, vector: k, normStats: k, embeddingAccumulate: k}
	d := &VulkanResidentDriver{Operator: op}
	ctx := context.Background()
	buffers := make([]train.ResidentBuffer, 4)
	for i, n := range []int{8, 6, 10, 8} {
		var err error
		buffers[i], err = d.NewBuffer(ctx, n)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := d.CompactEmbeddingAccumulate(ctx, buffers[3], buffers[0], buffers[1], buffers[2], 3, 2, 4, .25); err != nil {
		t.Fatal(err)
	}
	for _, x := range buffers {
		if x.Close() != nil {
			t.Fatal("close")
		}
	}
	for i, n := range []int{8, 6, 10, 8} {
		var err error
		buffers[i], err = d.NewBuffer(ctx, n)
		if err != nil {
			t.Fatal(err)
		}
	}
	mock.failure = "submit"
	if d.CompactEmbeddingAccumulate(ctx, buffers[3], buffers[0], buffers[1], buffers[2], 3, 2, 4, .25) == nil {
		t.Fatal("submit")
	}
	if d.terminal == nil || len(d.retained) != 1 || d.Close() == nil {
		t.Fatal("uncertain owner")
	}
	if d.CompactEmbeddingAccumulate(ctx, buffers[3], buffers[0], buffers[1], buffers[2], 3, 2, 4, .25) == nil {
		t.Fatal("retry")
	}
}
