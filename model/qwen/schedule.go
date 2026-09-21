package qwen

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
	"sort"
)

// LayerSchedulePlan is a lightweight, backend-neutral planning view over the
// current Qwen GPU residency state. It does not execute layers; it describes
// where a scheduler should focus next.
type LayerSchedulePlan struct {
	CompletePrefixLayers int                    `json:"complete_prefix_layers"`
	FirstOverflowLayer   int                    `json:"first_overflow_layer"`
	TotalLayers          int                    `json:"total_layers"`
	ResidentBytes        int64                  `json:"resident_bytes"`
	MissingBytes         int64                  `json:"missing_bytes"`
	TransientBytes       int64                  `json:"transient_bytes,omitempty"`
	TransientUploads     int64                  `json:"transient_uploads,omitempty"`
	Recommended          LayerWindowCandidate   `json:"recommended,omitempty"`
	BestFeasible         LayerWindowCandidate   `json:"best_feasible,omitempty"`
	Candidates           []LayerWindowCandidate `json:"candidates,omitempty"`
}

type LayerWindowCandidate struct {
	StartLayer               int     `json:"start_layer"`
	Layers                   int     `json:"layers"`
	MissingBytes             int64   `json:"missing_bytes"`
	ResidentBytes            int64   `json:"resident_bytes"`
	TotalBytes               int64   `json:"total_bytes"`
	TransientBytes           int64   `json:"transient_bytes,omitempty"`
	TransientUploads         int64   `json:"transient_uploads,omitempty"`
	EstimatedBytesPerToken   float64 `json:"estimated_bytes_per_token,omitempty"`
	EstimatedUploadsPerToken float64 `json:"estimated_uploads_per_token,omitempty"`
	FitsWindowBudget         bool    `json:"fits_window_budget"`
	FitsFreeMemory           bool    `json:"fits_free_memory"`
	Score                    float64 `json:"score,omitempty"`
}

// BuildLayerSchedulePlan turns observed GPU cache/residency stats into a small
// layer-window planning table. The scoring intentionally prefers windows that
// cover observed transient-heavy layers while fitting either the configured
// window budget or currently free VRAM.
func BuildLayerSchedulePlan(stats Qwen35GPUCacheStats, candidateSizes []int) LayerSchedulePlan {
	if len(candidateSizes) == 0 {
		candidateSizes = []int{2, 4, 8}
	}
	layers := make([]Qwen35GPULayerStat, 0, len(stats.MLXLayers))
	for _, l := range stats.MLXLayers {
		if l.Layer >= 0 && l.ResidentBytes >= 0 && l.TotalBytes >= l.ResidentBytes {
			layers = append(layers, l)
		}
	}
	sort.Slice(layers, func(i, j int) bool { return layers[i].Layer < layers[j].Layer })
	firstOverflow := stats.MLXCompletePrefixLayers
	if firstOverflow < 0 {
		firstOverflow = 0
	}
	plan := LayerSchedulePlan{CompletePrefixLayers: stats.MLXCompletePrefixLayers, FirstOverflowLayer: firstOverflow, TotalLayers: len(layers), TransientBytes: stats.TransientBytes, TransientUploads: stats.Transient}
	for _, l := range layers {
		plan.ResidentBytes = checked.SaturatingAddInt64(plan.ResidentBytes, l.ResidentBytes)
		if l.TotalBytes > l.ResidentBytes {
			plan.MissingBytes = checked.SaturatingAddInt64(plan.MissingBytes, l.TotalBytes-l.ResidentBytes)
		}
	}
	byLayer := map[int]Qwen35GPULayerStat{}
	for _, l := range layers {
		byLayer[l.Layer] = l
	}
	transientByLayer := map[int]Qwen35GPULayerStat{}
	for _, l := range stats.TransientLayers {
		if l.Layer >= 0 && l.Bytes >= 0 && l.Count >= 0 {
			transientByLayer[l.Layer] = l
		}
	}
	for _, n := range candidateSizes {
		if n <= 0 {
			continue
		}
		cand := LayerWindowCandidate{StartLayer: firstOverflow, Layers: n}
		// Work is bounded by supplied inventory, never by an arbitrary window.
		for _, l := range byLayer {
			if l.Layer < firstOverflow || l.Layer-firstOverflow >= n {
				continue
			}
			cand.ResidentBytes = checked.SaturatingAddInt64(cand.ResidentBytes, l.ResidentBytes)
			cand.TotalBytes = checked.SaturatingAddInt64(cand.TotalBytes, l.TotalBytes)
			if l.TotalBytes > l.ResidentBytes {
				cand.MissingBytes = checked.SaturatingAddInt64(cand.MissingBytes, l.TotalBytes-l.ResidentBytes)
			}
			if t, ok := transientByLayer[l.Layer]; ok {
				cand.TransientBytes = checked.SaturatingAddInt64(cand.TransientBytes, t.Bytes)
				cand.TransientUploads = checked.SaturatingAddInt64(cand.TransientUploads, t.Count)
			}
		}
		if cand.TotalBytes == 0 {
			continue
		}
		budget := stats.WindowBudgetBytes
		if budget <= 0 {
			if stats.FreeBytes > math.MaxInt64 {
				budget = math.MaxInt64
			} else {
				budget = int64(stats.FreeBytes)
			}
		}
		cand.FitsWindowBudget = budget > 0 && cand.MissingBytes <= budget
		cand.FitsFreeMemory = stats.FreeBytes > 0 && uint64(cand.MissingBytes) <= stats.FreeBytes
		if n > 0 {
			cand.EstimatedBytesPerToken = float64(cand.MissingBytes) / float64(n)
			if cand.TransientUploads > 0 {
				cand.EstimatedUploadsPerToken = float64(cand.TransientUploads) / float64(n)
			}
		}
		cand.Score = float64(cand.TransientBytes) - float64(cand.MissingBytes)*0.25
		if !cand.FitsWindowBudget && !cand.FitsFreeMemory {
			cand.Score *= 0.25
		}
		plan.Candidates = append(plan.Candidates, cand)
		if plan.Recommended.Layers == 0 || cand.Score > plan.Recommended.Score {
			plan.Recommended = cand
		}
		if cand.FitsWindowBudget || cand.FitsFreeMemory {
			if plan.BestFeasible.Layers == 0 || cand.Score > plan.BestFeasible.Score {
				plan.BestFeasible = cand
			}
		}
	}
	return plan
}
