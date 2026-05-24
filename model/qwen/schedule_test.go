package qwen

import "testing"

func TestBuildLayerSchedulePlan(t *testing.T) {
	stats := Qwen35GPUCacheStats{
		WindowBudgetBytes:       500,
		FreeBytes:               400,
		MLXCompletePrefixLayers: 2,
		TransientBytes:          1000,
		Transient:               10,
		MLXLayers: []Qwen35GPULayerStat{
			{Layer: 0, ResidentBytes: 100, TotalBytes: 100},
			{Layer: 1, ResidentBytes: 100, TotalBytes: 100},
			{Layer: 2, ResidentBytes: 40, TotalBytes: 200},
			{Layer: 3, ResidentBytes: 0, TotalBytes: 200},
			{Layer: 4, ResidentBytes: 0, TotalBytes: 300},
		},
		TransientLayers: []Qwen35GPULayerStat{
			{Layer: 2, Count: 4, Bytes: 800},
			{Layer: 3, Count: 4, Bytes: 900},
			{Layer: 4, Count: 2, Bytes: 100},
		},
	}
	plan := BuildLayerSchedulePlan(stats, []int{1, 2})
	if plan.FirstOverflowLayer != 2 || plan.CompletePrefixLayers != 2 || len(plan.Candidates) != 2 {
		t.Fatalf("bad plan: %+v", plan)
	}
	c := plan.Candidates[1]
	if c.StartLayer != 2 || c.Layers != 2 || c.MissingBytes != 360 || c.TransientBytes != 1700 || !c.FitsWindowBudget || !c.FitsFreeMemory {
		t.Fatalf("bad 2-layer candidate: %+v", c)
	}
	if plan.Recommended.Layers == 0 {
		t.Fatalf("missing recommendation: %+v", plan)
	}
}
