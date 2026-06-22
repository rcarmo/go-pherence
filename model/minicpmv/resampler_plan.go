package minicpmv

import (
	"sort"
	"strings"
)

type ResamplerTensorRole string

const (
	ResamplerQuery    ResamplerTensorRole = "query"
	ResamplerPosEmbed ResamplerTensorRole = "pos_embed"
	ResamplerKVProj   ResamplerTensorRole = "kv_proj"
	ResamplerQProj    ResamplerTensorRole = "q_proj"
	ResamplerKProj    ResamplerTensorRole = "k_proj"
	ResamplerVProj    ResamplerTensorRole = "v_proj"
	ResamplerOProj    ResamplerTensorRole = "o_proj"
	ResamplerNorm     ResamplerTensorRole = "norm"
	ResamplerMLP      ResamplerTensorRole = "mlp"
	ResamplerOther    ResamplerTensorRole = "other"
)

type ResamplerTensorBinding struct {
	Role ResamplerTensorRole `json:"role"`
	Name string              `json:"name"`
}

type ResamplerTensorPlan struct {
	Bindings          []ResamplerTensorBinding    `json:"bindings"`
	Counts            map[ResamplerTensorRole]int `json:"counts"`
	MissingRequired   []ResamplerTensorRole       `json:"missing_required,omitempty"`
	Ready             bool                        `json:"ready"`
	NeedsKVProjection bool                        `json:"needs_kv_projection"`
}

func BuildResamplerTensorPlan(names []string, needsKVProjection bool) ResamplerTensorPlan {
	plan := ResamplerTensorPlan{Counts: map[ResamplerTensorRole]int{}, NeedsKVProjection: needsKVProjection}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	for _, name := range sorted {
		if ClassifyTensorName(name) != TensorResampler && ClassifyTensorName(name) != TensorProjector {
			continue
		}
		role := ClassifyResamplerTensorName(name)
		plan.Bindings = append(plan.Bindings, ResamplerTensorBinding{Role: role, Name: name})
		plan.Counts[role]++
	}
	required := []ResamplerTensorRole{ResamplerQuery}
	if needsKVProjection {
		required = append(required, ResamplerKVProj)
	}
	for _, role := range required {
		if plan.Counts[role] == 0 {
			plan.MissingRequired = append(plan.MissingRequired, role)
		}
	}
	plan.Ready = len(plan.MissingRequired) == 0 && len(plan.Bindings) > 0
	return plan
}

func ClassifyResamplerTensorName(name string) ResamplerTensorRole {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "query") || strings.Contains(n, "query_tokens"):
		return ResamplerQuery
	case strings.Contains(n, "pos_embed") || strings.Contains(n, "position"):
		return ResamplerPosEmbed
	case strings.Contains(n, "kv_proj") || strings.Contains(n, "vision_proj") || strings.Contains(n, "mm_projector") || strings.Contains(n, "multi_modal_projector"):
		return ResamplerKVProj
	case strings.Contains(n, "q_proj"):
		return ResamplerQProj
	case strings.Contains(n, "k_proj"):
		return ResamplerKProj
	case strings.Contains(n, "v_proj"):
		return ResamplerVProj
	case strings.Contains(n, "o_proj") || strings.Contains(n, "out_proj"):
		return ResamplerOProj
	case strings.Contains(n, "norm") || strings.Contains(n, "ln"):
		return ResamplerNorm
	case strings.Contains(n, "mlp") || strings.Contains(n, "fc") || strings.Contains(n, "gate_proj") || strings.Contains(n, "up_proj") || strings.Contains(n, "down_proj"):
		return ResamplerMLP
	default:
		return ResamplerOther
	}
}
