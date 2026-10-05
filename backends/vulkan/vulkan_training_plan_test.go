package vulkan

import (
	"context"
	"testing"
)

func TestVulkanOfflineTrainingAttentionPlan(t *testing.T) {
	_, kernel, _, _ := newPlanMock(t)
	kernel.numBuffers = 7
	kernel.pushSize = 24
	op := &VkTrainingF32{attention: kernel}
	tensors := make([]*VkTensorF32, 10)
	for i := range tensors {
		tensors[i] = linearMetadataTensor(10+i, 3, 2, 2)
	}
	for _, i := range []int{3, 4, 8} {
		tensors[i] = linearMetadataTensor(10+i, 2, 3, 3)
	}
	dq, dk, dv, dp, ds, q, k, v, p, dy := tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7], tensors[8], tensors[9]
	plan, err := op.AttentionBackwardPlan(context.Background(), dq, dk, dv, dp, ds, q, k, v, p, dy, 2, .5)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.state.stages) != 5 {
		t.Fatal("stage count")
	}
	for i, stage := range plan.state.stages {
		if stage.push[4] != uint32(i) {
			t.Fatal("dependency stage order")
		}
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []struct{ dk, dp *VkTensorF32 }{{dq, dp}, {dk, q}, {dk, ds}} {
		if _, err := op.AttentionBackwardPlan(context.Background(), dq, bad.dk, dv, bad.dp, ds, q, k, v, p, dy, 2, .5); err == nil {
			t.Fatal("composite alias admitted")
		}
	}
}
func TestVulkanOfflineTrainingNormPlan(t *testing.T) {
	_, kernel, _, _ := newPlanMock(t)
	kernel.numBuffers = 5
	kernel.pushSize = 12
	op := &VkTrainingF32{norm: kernel}
	dx, dw, db := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 5), linearMetadataTensor(3, 5)
	x, inv, w, dy := linearMetadataTensor(4, 3, 5), linearMetadataTensor(5, 3), linearMetadataTensor(6, 5), linearMetadataTensor(7, 3, 5)
	plan, err := op.LayerNormBackwardPlan(context.Background(), dx, dw, db, x, inv, w, dy)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.state.stages) != 3 {
		t.Fatal("stage count")
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := op.LayerNormBackwardPlan(context.Background(), dx, dw, dw, x, inv, w, dy); err == nil {
		t.Fatal("parameter derivative alias")
	}
}
