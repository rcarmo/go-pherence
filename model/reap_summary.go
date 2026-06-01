package model

// REAPSummary is a compact readiness/diagnostic view of static expert pruning.
type REAPSummary struct {
	Enabled          bool    `json:"enabled"`
	PruneRatio       float64 `json:"prune_ratio,omitempty"`
	Source           string  `json:"source,omitempty"`
	DefaultExperts   int     `json:"default_experts,omitempty"`
	LayerMasks       int     `json:"layer_masks,omitempty"`
	LayerExpertTotal int     `json:"layer_expert_total,omitempty"`
	HasStaticMasks   bool    `json:"has_static_masks"`
}

func (r *REAPConfig) Summary() REAPSummary {
	if r == nil || !r.Enabled {
		return REAPSummary{}
	}
	out := REAPSummary{Enabled: true, PruneRatio: r.PruneRatio, Source: r.Source, DefaultExperts: len(r.DefaultMask), LayerMasks: len(r.LayerActiveNumeric)}
	for _, mask := range r.LayerActiveNumeric {
		out.LayerExpertTotal += len(mask)
	}
	out.HasStaticMasks = out.DefaultExperts > 0 || out.LayerMasks > 0
	return out
}
