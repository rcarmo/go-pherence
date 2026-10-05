package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingDeviceScale(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingAdamWScaled)
	if err != nil || c.StorageBindings != 63 || c.PushBytes != 40 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers, k.pushSize = 6, 40
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{adamScaled: k}
	b := make([]*VkTensorF32, 6)
	for i := range b {
		n := 257
		if i == 5 {
			n = 4
		}
		b[i] = linearMetadataTensor(i+1, n)
	}
	p, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 3)
	stage := func(x []*VkTensorF32, q train.AdamWStep) (VkF32Stage, error) {
		return op.AdamWScaledStage(context.Background(), x[0], x[1], x[2], x[3], x[4], x[5], q)
	}
	s, err := stage(b, p)
	if err != nil || len(s.Tensors) != 6 || s.Tensors[5].Elements() != 4 || s.Groups != [3]uint32{2, 1, 1} || len(s.PushWords) != 10 || s.PushWords[9] != math.Float32bits(1) {
		t.Fatal("ABI", s, err)
	}
	for i := 0; i < 6; i++ {
		for j := 0; j < i; j++ {
			x := append([]*VkTensorF32(nil), b...)
			x[i] = x[j]
			if _, err = stage(x, p); err == nil {
				t.Fatal("alias", i, j)
			}
		}
	}
	for i := range b {
		x := append([]*VkTensorF32(nil), b...)
		x[i] = nil
		if _, err = stage(x, p); err == nil {
			t.Fatal("nil", i)
		}
	}
	q := p
	q.GradientScale = .5
	if _, err = stage(b, q); err == nil {
		t.Fatal("double scaling")
	}
	x := append([]*VkTensorF32(nil), b...)
	x[5] = linearMetadataTensor(6, 3)
	if _, err = stage(x, p); err == nil {
		t.Fatal("short clip")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = op.AdamWScaledStage(ctx, b[0], b[1], b[2], b[3], b[4], b[5], p); err == nil {
		t.Fatal("cancel")
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err = bad.AdamWScaledStage(context.Background(), b[0], b[1], b[2], b[3], b[4], b[5], p); err == nil {
			t.Fatal("owner")
		}
	}
}
func TestVulkanOfflineDeviceScaleDriverRetention(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	mock.failure = "submit"
	k.numBuffers, k.pushSize = 6, 40
	d := &VulkanResidentDriver{Operator: &VkTrainingF32{adamScaled: k}}
	b := make([]train.ResidentBuffer, 6)
	for i := range b {
		n := 2
		if i == 5 {
			n = 4
		}
		b[i] = &vkResidentBuffer{tensor: linearMetadataTensor(i+1, n)}
	}
	p, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, 0, 0, 1, 1)
	if err := d.ValidateScaledUpdate(context.Background(), b[0], b[1], b[2], b[3], b[4], b[5], 2, p); err != nil {
		t.Fatal(err)
	}
	if err := d.UpdateScaled(context.Background(), b[0], b[1], b[2], b[3], b[4], b[5], 2, p); err == nil || d.terminal == nil || len(d.retained) != 1 || d.Close() == nil {
		t.Fatal("submitted failure retention")
	}
	if err := d.UpdateScaled(context.Background(), b[0], b[1], b[2], b[3], b[4], b[5], 2, p); err == nil || len(d.retained) != 1 {
		t.Fatal("retry")
	}
}
