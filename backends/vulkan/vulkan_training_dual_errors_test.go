package vulkan

import (
	"context"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDualErrors(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{linearDual: kernel, normDual: kernel, modDual: kernel, normDualForward: kernel}
	tensor := func(i int, shape ...int) *VkTensorF32 { return linearMetadataTensor(i+1, shape...) }
	x, xt, w, gv, gt, dst, stats := tensor(0, 3, 5), tensor(1, 3, 5), tensor(2, 5), tensor(3, 3, 5), tensor(4, 3, 5), tensor(5, 3, 5), tensor(6, 3, 5)
	if _, err := (*VkTrainingF32)(nil).NormDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, stats, 0, 0); err == nil {
		t.Fatal("nil operator")
	}
	kernel.numBuffers = 7
	kernel.pushSize = 16
	original := xt.shape
	xt.shape[0] = 2
	if _, err := op.NormDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, stats, 0, 0); err == nil {
		t.Fatal("norm shape")
	}
	xt.shape = original
	dst.rank = 1
	if _, err := op.NormDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, stats, 0, 0); err == nil {
		t.Fatal("norm dst")
	}
	dst.rank = 2
	if _, err := op.NormDualBackwardStage(context.Background(), dst, x, xt, w, gv, gt, stats, 2, 0); err == nil {
		t.Fatal("kind")
	}
	kernel.numBuffers = 6
	kernel.pushSize = 16
	linearW := tensor(7, 5, 5)
	if _, err := op.LinearDualBackwardStage(context.Background(), dst, x, xt, linearW, gv, gt, 4); err == nil {
		t.Fatal("mode")
	}
	linearW.shape[1] = 4
	if _, err := op.LinearDualBackwardStage(context.Background(), dst, x, xt, linearW, gv, gt, 0); err == nil {
		t.Fatal("linear input shape")
	}
	linearW.shape[1] = 5
	if _, err := op.LinearDualBackwardStage(context.Background(), tensor(8, 4, 5), x, xt, linearW, gv, gt, 0); err == nil {
		t.Fatal("linear dst shape")
	}
	if _, err := op.LinearDualBackwardStage(context.Background(), tensor(8, 4, 5), x, xt, linearW, gv, gt, 2); err == nil {
		t.Fatal("linear weight shape")
	}
	if _, err := op.LinearDualBackwardStage(context.Background(), tensor(8, 6), x, xt, linearW, gv, gt, 3); err == nil {
		t.Fatal("linear bias shape")
	}
	kernel.pushSize = 20
	if _, err := op.NormDualForwardStage(context.Background(), tensor(9, 3, 4), x, xt, w, w, stats, 0, 0, 1e-4); err == nil {
		t.Fatal("stats shape")
	}
	if _, err := op.NormDualForwardStage(context.Background(), tensor(9, 3, 6), x, xt, w, w, stats, 0, 1, 1e-4); err == nil {
		t.Fatal("forward shape")
	}
	stats.shape[1] = 4
	if _, err := op.NormDualForwardStage(context.Background(), dst, x, xt, w, w, stats, 0, 1, 1e-4); err == nil {
		t.Fatal("forward stats shape")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.NormDualForwardStage(ctx, dst, x, xt, w, w, stats, 0, 1, 1e-4); err == nil {
		t.Fatal("cancel")
	}
	kernel.numBuffers = 7
	kernel.pushSize = 12
	if _, err := op.ModDualBackwardStage(context.Background(), tensor(10, 3), tensor(0, 3), tensor(1, 3), tensor(2, 5), tensor(3, 6), tensor(4, 3), tensor(5, 3), 0, 0); err == nil {
		t.Fatal("mod shape")
	}
}
