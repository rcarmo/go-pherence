package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildSlicePlanDisabled(t *testing.T) {
	plan := BuildSlicePlan(config.MiniCPMVSummary{SliceMode: false, PatchSize: 14}, 1024, 768)
	if !plan.Ready || plan.Enabled || plan.EstimatedSlices != 1 {
		t.Fatalf("bad disabled slice plan: %+v", plan)
	}
}

func TestBuildSlicePlanEnabled(t *testing.T) {
	summary := config.MiniCPMVSummary{SliceMode: true, ScaleResolution: 448, SlicePatchSize: 14, MaxSliceNums: 9}
	plan := BuildSlicePlan(summary, 1024, 900)
	if !plan.Ready || !plan.Enabled || plan.EstimatedSlices != 9 || plan.PatchSize != 14 {
		t.Fatalf("bad slice plan: %+v", plan)
	}
}

func TestBuildSlicePlanCapsMaxSlices(t *testing.T) {
	summary := config.MiniCPMVSummary{SliceMode: true, ScaleResolution: 448, PatchSize: 14, MaxSliceNums: 4}
	plan := BuildSlicePlan(summary, 2000, 2000)
	if !plan.Ready || plan.EstimatedSlices != 4 {
		t.Fatalf("expected capped slices: %+v", plan)
	}
}

func TestBuildSlicePlanMissingMetadata(t *testing.T) {
	plan := BuildSlicePlan(config.MiniCPMVSummary{SliceMode: true}, 100, 100)
	if plan.Ready || plan.EstimatedSlices != 0 {
		t.Fatalf("expected not ready without scale/patch metadata: %+v", plan)
	}
}
