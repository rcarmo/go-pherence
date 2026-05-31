package gguf

import "strings"

// Inspection is a lightweight GGUF readiness summary. It reads only metadata and
// tensor index data, so it is safe for large local checkpoints.
type Inspection struct {
	Path              string         `json:"path"`
	Architecture      string         `json:"architecture,omitempty"`
	Name              string         `json:"name,omitempty"`
	TensorCount       int            `json:"tensor_count"`
	QuantCounts       map[string]int `json:"quant_counts"`
	HasQ4K            bool           `json:"has_q4_k"`
	HasMoE            bool           `json:"has_moe"`
	Experts           uint32         `json:"experts,omitempty"`
	ExpertsPerToken   uint32         `json:"experts_per_token,omitempty"`
	HasREAPMetadata   bool           `json:"has_reap_metadata"`
	REAPMetadataKeys  []string       `json:"reap_metadata_keys,omitempty"`
	TurboQuantReady   bool           `json:"turboquant_ready"`
	PureGoSIMDReady   bool           `json:"pure_go_simd_ready"`
	ReadinessWarnings []string       `json:"readiness_warnings,omitempty"`
}

func Inspect(path string) (Inspection, error) {
	g, err := Open(path)
	if err != nil {
		return Inspection{}, err
	}
	defer g.Close()
	return InspectOpen(path, g), nil
}

func InspectOpen(path string, g *GGUF) Inspection {
	in := Inspection{Path: path, TensorCount: len(g.Tensors), QuantCounts: make(map[string]int)}
	if arch, ok := g.MetaString("general.architecture"); ok {
		in.Architecture = arch
	}
	if name, ok := g.MetaString("general.name"); ok {
		in.Name = name
	}
	for _, t := range g.Tensors {
		name := quantTypeName(t.QType)
		in.QuantCounts[name]++
		if t.QType == QuantQ4_K {
			in.HasQ4K = true
		}
		if strings.Contains(t.Name, "ffn_gate_exps") || strings.Contains(t.Name, "ffn_up_exps") || strings.Contains(t.Name, "ffn_down_exps") || strings.Contains(t.Name, "block_sparse_moe") {
			in.HasMoE = true
		}
	}
	prefixes := []string{in.Architecture, "llama", "qwen3moe", "qwen3", "qwen2moe", "qwen2"}
	for _, p := range prefixes {
		if p == "" {
			continue
		}
		if v, ok := g.MetaUint32(p + ".expert_count"); ok {
			in.Experts = v
			in.HasMoE = true
		}
		if v, ok := g.MetaUint32(p + ".expert_used_count"); ok {
			in.ExpertsPerToken = v
			in.HasMoE = true
		}
	}
	for k := range g.Meta {
		lk := strings.ToLower(k)
		if strings.Contains(lk, "reap") || strings.Contains(lk, "prun") {
			in.HasREAPMetadata = true
			in.REAPMetadataKeys = append(in.REAPMetadataKeys, k)
		}
	}
	in.TurboQuantReady = true
	in.PureGoSIMDReady = in.TensorCount > 0 && (in.HasQ4K || len(in.QuantCounts) > 0)
	if in.Architecture == "" {
		in.ReadinessWarnings = append(in.ReadinessWarnings, "missing general.architecture metadata")
	}
	if in.HasMoE && in.Experts == 0 {
		in.ReadinessWarnings = append(in.ReadinessWarnings, "MoE tensors found but expert_count metadata was not detected")
	}
	return in
}

func quantTypeName(q QuantType) string {
	switch q {
	case QuantF32:
		return "F32"
	case QuantF16:
		return "F16"
	case QuantQ4_0:
		return "Q4_0"
	case QuantQ4_1:
		return "Q4_1"
	case QuantQ8_0:
		return "Q8_0"
	case QuantQ2_K:
		return "Q2_K"
	case QuantQ3_K:
		return "Q3_K"
	case QuantQ4_K:
		return "Q4_K"
	case QuantQ5_K:
		return "Q5_K"
	case QuantQ6_K:
		return "Q6_K"
	case QuantQ8_K:
		return "Q8_K"
	default:
		return "UNKNOWN"
	}
}
