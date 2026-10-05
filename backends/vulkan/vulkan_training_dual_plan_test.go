package vulkan

import (
	"context"
	"testing"
)

func TestVulkanOfflineTrainingDualPlans(t *testing.T) {
	for _, norm := range []bool{false, true} {
		t.Run(map[bool]string{false: "linear", true: "norm"}[norm], func(t *testing.T) {
			_, kernel, _, _ := newPlanMock(t)
			kernel.pushSize = 16
			kernel.numBuffers = 6
			if norm {
				kernel.numBuffers = 7
			}
			op := &VkTrainingF32{linearDual: kernel, normDual: kernel}
			x, xt, w, gv, gt := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 3, 5), linearMetadataTensor(3, 7, 5), linearMetadataTensor(4, 3, 7), linearMetadataTensor(5, 3, 7)
			dx, dt, dw, db := linearMetadataTensor(6, 3, 5), linearMetadataTensor(7, 3, 5), linearMetadataTensor(8, 7, 5), linearMetadataTensor(9, 7)
			var plan *VkF32Plan
			var err error
			if norm {
				w = linearMetadataTensor(3, 5)
				gv = linearMetadataTensor(4, 3, 5)
				gt = linearMetadataTensor(5, 3, 5)
				dw = linearMetadataTensor(8, 5)
				db = linearMetadataTensor(9, 5)
				stats := linearMetadataTensor(10, 3, 5)
				plan, err = op.NormDualBackwardPlan(context.Background(), dx, dt, dw, db, x, xt, w, gv, gt, stats, 0)
				if _, err := op.NormDualBackwardPlan(context.Background(), dx, dx, dw, db, x, xt, w, gv, gt, stats, 0); err == nil {
					t.Fatal("dual composite alias")
				}
				if _, err := op.NormDualBackwardPlan(context.Background(), dx, dt, dw, db, x, xt, w, gv, gt, stats, 1); err == nil {
					t.Fatal("variance bias")
				}
			} else {
				plan, err = op.LinearDualBackwardPlan(context.Background(), dx, dt, dw, db, x, xt, w, gv, gt)
				if _, err := op.LinearDualBackwardPlan(context.Background(), dx, dx, dw, db, x, xt, w, gv, gt); err == nil {
					t.Fatal("dual composite alias")
				}
			}
			if err != nil || len(plan.state.stages) != 4 {
				t.Fatal(plan, err)
			}
			if err := plan.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
