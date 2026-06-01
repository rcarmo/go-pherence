package gguf

import "strings"

// Inspection is a lightweight GGUF readiness summary. It reads only metadata and
// tensor index data, so it is safe for large local checkpoints.
type Inspection struct {
	Path                  string         `json:"path"`
	Architecture          string         `json:"architecture,omitempty"`
	Name                  string         `json:"name,omitempty"`
	TensorCount           int            `json:"tensor_count"`
	QuantCounts           map[string]int `json:"quant_counts"`
	HasQ4K                bool           `json:"has_q4_k"`
	HasMoE                bool           `json:"has_moe"`
	Experts               uint32         `json:"experts,omitempty"`
	ExpertsPerToken       uint32         `json:"experts_per_token,omitempty"`
	HasREAPMetadata       bool           `json:"has_reap_metadata"`
	REAPMetadataKeys      []string       `json:"reap_metadata_keys,omitempty"`
	TurboQuantReady       bool           `json:"turboquant_ready"`
	PureGoSIMDReady       bool           `json:"pure_go_simd_ready"`
	RuntimeSupported      bool           `json:"runtime_supported"`
	MissingRuntimeTensors []string       `json:"missing_runtime_tensors,omitempty"`
	ReadinessWarnings     []string       `json:"readiness_warnings,omitempty"`
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
	if !in.HasREAPMetadata && strings.Contains(strings.ToLower(path+" "+in.Name), "reap") {
		in.HasREAPMetadata = true
		in.REAPMetadataKeys = append(in.REAPMetadataKeys, "filename_or_name")
	}
	in.TurboQuantReady = true
	in.PureGoSIMDReady = in.TensorCount > 0 && (in.HasQ4K || len(in.QuantCounts) > 0)
	in.MissingRuntimeTensors = missingRuntimeTensors(g, in.Architecture, in.HasMoE)
	in.RuntimeSupported = in.PureGoSIMDReady && len(in.MissingRuntimeTensors) == 0
	if in.Architecture == "" {
		in.ReadinessWarnings = append(in.ReadinessWarnings, "missing general.architecture metadata")
	}
	if in.HasMoE && in.Experts == 0 {
		in.ReadinessWarnings = append(in.ReadinessWarnings, "MoE tensors found but expert_count metadata was not detected")
	}
	return in
}

func missingRuntimeTensors(g *GGUF, arch string, hasMoE bool) []string {
	if g == nil {
		return []string{"<nil gguf>"}
	}
	// Current GGUFLlama runtime expects llama.cpp split attention tensors. Newer
	// Qwen3.5/Qwen3.6 MoE GGUF files use fused attn_qkv plus SSM/hybrid blocks;
	// report that explicitly instead of claiming generation readiness from quant
	// metadata alone.
	required := []string{"token_embd.weight", "output_norm.weight", "output.weight"}
	if isQwenNextHybridGGUF(g, arch) {
		required = append(required,
			"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.post_attention_norm.weight",
			"blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight",
			"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
		)
		return missingTensorNames(g, required)
	}
	if hasMoE {
		required = append(required,
			"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
			"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
			"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
		)
	} else {
		required = append(required,
			"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
			"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		)
	}
	return missingTensorNames(g, required)
}

func missingTensorNames(g *GGUF, required []string) []string {
	var missing []string
	for _, name := range required {
		if _, ok := g.TensorByName(name); !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

func isQwenNextHybridGGUF(g *GGUF, arch string) bool {
	if g == nil {
		return false
	}
	keys := []string{arch + ".ssm.inner_size", arch + ".ssm.state_size", arch + ".full_attention_interval"}
	for _, key := range keys {
		if key == ".ssm.inner_size" || key == ".ssm.state_size" || key == ".full_attention_interval" {
			continue
		}
		if _, ok := g.MetaUint32(key); ok {
			return true
		}
	}
	if _, ok := g.TensorByName("blk.0.attn_qkv.weight"); ok {
		if _, ok := g.TensorByName("blk.0.ssm_out.weight"); ok {
			return true
		}
	}
	return false
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
