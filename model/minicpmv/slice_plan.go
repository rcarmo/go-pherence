package minicpmv

import "github.com/rcarmo/go-pherence/loader/config"

type SlicePlan struct {
	Enabled         bool `json:"enabled"`
	ImageWidth      int  `json:"image_width,omitempty"`
	ImageHeight     int  `json:"image_height,omitempty"`
	ScaleResolution int  `json:"scale_resolution,omitempty"`
	PatchSize       int  `json:"patch_size,omitempty"`
	MaxSliceNums    int  `json:"max_slice_nums,omitempty"`
	EstimatedSlices int  `json:"estimated_slices"`
	Ready           bool `json:"ready"`
}

func BuildSlicePlan(summary config.MiniCPMVSummary, imageWidth, imageHeight int) SlicePlan {
	patch := firstPositive(summary.SlicePatchSize, summary.PatchSize)
	plan := SlicePlan{Enabled: summary.SliceMode, ImageWidth: imageWidth, ImageHeight: imageHeight, ScaleResolution: summary.ScaleResolution, PatchSize: patch, MaxSliceNums: summary.MaxSliceNums}
	if !summary.SliceMode {
		plan.EstimatedSlices = 1
		plan.Ready = true
		return plan
	}
	if imageWidth <= 0 || imageHeight <= 0 || summary.ScaleResolution <= 0 || patch <= 0 {
		return plan
	}
	tilesX := ceilDiv(imageWidth, summary.ScaleResolution)
	tilesY := ceilDiv(imageHeight, summary.ScaleResolution)
	slices := tilesX * tilesY
	if slices < 1 {
		slices = 1
	}
	if summary.MaxSliceNums > 0 && slices > summary.MaxSliceNums {
		slices = summary.MaxSliceNums
	}
	plan.EstimatedSlices = slices
	plan.Ready = true
	return plan
}

func ceilDiv(a, b int) int {
	if a <= 0 || b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}
