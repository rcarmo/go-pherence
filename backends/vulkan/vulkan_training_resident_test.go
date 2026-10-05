package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineResidentNorm(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingNormStats)
	if err != nil || c.StorageBindings != 3 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 2
	kernel.pushSize = 8
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{normStats: kernel}
	x, dst := linearMetadataTensor(1, 257), linearMetadataTensor(2, 6)
	for _, positive := range []bool{false, true} {
		s, err := op.NormStatsStage(context.Background(), dst, x, positive)
		flag := uint32(0)
		if positive {
			flag = 1
		}
		if err != nil || s.Groups != [3]uint32{1, 1, 1} || s.PushWords[0] != 257 || s.PushWords[1] != flag {
			t.Fatal(s, err)
		}
	}
	for _, inputs := range [][2]*VkTensorF32{{nil, x}, {dst, nil}, {x, x}, {linearMetadataTensor(3, 5), x}, {dst, linearMetadataTensor(4, 1, 257)}} {
		if _, err := op.NormStatsStage(context.Background(), inputs[0], inputs[1], false); err == nil {
			t.Fatal("invalid norm tensor")
		}
	}
	var nilOp *VkTrainingF32
	if _, err := nilOp.NormStatsStage(context.Background(), dst, x, false); err == nil {
		t.Fatal("nil owner")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.NormStatsStage(ctx, dst, x, false); err == nil {
		t.Fatal("cancel")
	}
}
func TestVulkanOfflineResidentDriver(t *testing.T) {
	ctx := context.Background()
	d := &VulkanResidentDriver{Operator: &VkTrainingF32{}}
	if _, err := d.NewBuffer(ctx, 2); err == nil {
		t.Fatal("unowned kernels")
	}
	if _, err := vkResidentBuffers(nil); err == nil {
		t.Fatal("nil tensor")
	}
	for _, bad := range []*VulkanResidentDriver{nil, {}} {
		if bad.Accumulate(ctx, nil, nil, nil, 1, 1) == nil || bad.NormStats(ctx, nil, nil, 1, false) == nil || bad.ValidateUpdate(ctx, nil, nil, nil, nil, nil, 1, train.AdamWStep{}) == nil {
			t.Fatal("nil driver")
		}
	}
	// Host metadata admission does not create a runtime or native allocation.
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 5
	kernel.pushSize = 40
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	d.Operator.adam = kernel
	b := make([]train.ResidentBuffer, 5)
	for i := range b {
		b[i] = &vkResidentBuffer{tensor: linearMetadataTensor(i+1, 2)}
	}
	step, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 1)
	if err := d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 2, step); err != nil {
		t.Fatal(err)
	}
	if d.ValidateUpdate(ctx, b[0], b[0], b[2], b[3], b[4], 2, step) == nil || d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 3, step) == nil {
		t.Fatal("invalid range")
	}
	if d.Close() != nil || (*VulkanResidentDriver)(nil).Close() != nil {
		t.Fatal("close")
	}
	d.terminal = context.Canceled
	if d.Close() == nil {
		t.Fatal("terminal plans dropped")
	}
}

func TestVulkanOfflineResidentTransfers(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	op := &VkTrainingF32{adam: k, vector: k, normStats: k}
	d := &VulkanResidentDriver{Operator: op}
	ctx := context.Background()
	b := make([]train.ResidentBuffer, 5)
	for i := range b {
		var err error
		b[i], err = d.NewBuffer(ctx, 2)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := b[0].Upload(ctx, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	x := make([]float32, 2)
	if err := b[0].Download(ctx, x); err != nil {
		t.Fatal(err)
	}
	if err := d.Accumulate(ctx, b[2], b[0], b[1], 2, 1); err != nil {
		t.Fatal(err)
	}
	stats, err := d.NewBuffer(ctx, 3)
	if err != nil {
		t.Fatal(err)
	}
	k.numBuffers, k.pushSize = 2, 8
	if err := d.NormStats(ctx, stats, b[0], 2, false); err != nil {
		t.Fatal(err)
	}
	k.numBuffers, k.pushSize = 5, 40
	step, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 1)
	if err := d.Update(ctx, b[0], b[1], b[2], b[3], b[4], 2, step); err != nil {
		t.Fatal(err)
	}
	for _, buffer := range append(b, stats) {
		if err := buffer.Close(); err != nil {
			t.Fatal(err)
		}
	}
	// Launch uncertainty retains the plan and blocks Close/retry. Mock runtime
	// functions only; the test does not open or execute a native Vulkan device.
	k.numBuffers, k.pushSize = 3, 20
	a, err := d.NewBuffer(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	bb, err := d.NewBuffer(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	o, err := d.NewBuffer(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	mock.failure = "submit"
	if err := d.Accumulate(ctx, o, a, bb, 2, 1); err == nil {
		t.Fatal("failed submit accepted")
	}
	if d.terminal == nil || len(d.retained) != 1 || d.Close() == nil {
		t.Fatal("uncertain plan owner dropped")
	}
	if err := d.Accumulate(ctx, o, a, bb, 2, 1); err == nil {
		t.Fatal("uncertain retry")
	}
}
