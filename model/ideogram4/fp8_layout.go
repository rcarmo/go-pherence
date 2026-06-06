package ideogram4

import (
	"fmt"
	"strconv"
	"strings"
)

type LinearRole string

const (
	RoleUnknown      LinearRole = "unknown"
	RoleInputProj    LinearRole = "input_proj"
	RoleLLMCondProj  LinearRole = "llm_cond_proj"
	RoleTimeEmbedIn  LinearRole = "time_embed_in"
	RoleTimeEmbedOut LinearRole = "time_embed_out"
	RoleAdaLNProj    LinearRole = "adaln_proj"
	RoleLayerAdaLN   LinearRole = "layer_adaln"
	RoleAttentionQKV LinearRole = "attention_qkv"
	RoleAttentionO   LinearRole = "attention_o"
	RoleMLPW1        LinearRole = "mlp_w1"
	RoleMLPW2        LinearRole = "mlp_w2"
	RoleMLPW3        LinearRole = "mlp_w3"
	RoleFinalAdaLN   LinearRole = "final_adaln"
	RoleFinalLinear  LinearRole = "final_linear"
)

type LinearSpec struct {
	Prefix      string     `json:"prefix"`
	Role        LinearRole `json:"role"`
	Layer       int        `json:"layer,omitempty"`
	InDim       int        `json:"in_dim"`
	OutDim      int        `json:"out_dim"`
	Weight      string     `json:"weight"`
	WeightScale string     `json:"weight_scale"`
}

func LinearSpecForPrefix(cfg Config, prefix string) (LinearSpec, bool) {
	if err := cfg.Validate(); err != nil || prefix == "" {
		return LinearSpec{}, false
	}
	s := LinearSpec{Prefix: prefix, Layer: -1, Weight: prefix + ".weight", WeightScale: prefix + ".weight_scale"}
	switch prefix {
	case "input_proj":
		s.Role, s.InDim, s.OutDim = RoleInputProj, cfg.InChannels, cfg.EmbDim
	case "llm_cond_proj":
		s.Role, s.InDim, s.OutDim = RoleLLMCondProj, cfg.LLMFeaturesDim, cfg.EmbDim
	case "t_embedding.mlp_in":
		s.Role, s.InDim, s.OutDim = RoleTimeEmbedIn, cfg.EmbDim, cfg.EmbDim
	case "t_embedding.mlp_out":
		s.Role, s.InDim, s.OutDim = RoleTimeEmbedOut, cfg.EmbDim, cfg.EmbDim
	case "adaln_proj":
		s.Role, s.InDim, s.OutDim = RoleAdaLNProj, cfg.EmbDim, cfg.AdaLNDim
	case "final_layer.adaln_modulation":
		s.Role, s.InDim, s.OutDim = RoleFinalAdaLN, cfg.AdaLNDim, cfg.EmbDim
	case "final_layer.linear":
		s.Role, s.InDim, s.OutDim = RoleFinalLinear, cfg.EmbDim, cfg.InChannels
	default:
		layer, suffix, ok := splitLayerPrefix(prefix)
		if !ok || layer < 0 || layer >= cfg.NumLayers {
			return LinearSpec{}, false
		}
		s.Layer = layer
		switch suffix {
		case "adaln_modulation":
			s.Role, s.InDim, s.OutDim = RoleLayerAdaLN, cfg.AdaLNDim, 4*cfg.EmbDim
		case "attention.qkv":
			s.Role, s.InDim, s.OutDim = RoleAttentionQKV, cfg.EmbDim, 3*cfg.EmbDim
		case "attention.o":
			s.Role, s.InDim, s.OutDim = RoleAttentionO, cfg.EmbDim, cfg.EmbDim
		case "feed_forward.w1":
			s.Role, s.InDim, s.OutDim = RoleMLPW1, cfg.EmbDim, cfg.IntermediateSize
		case "feed_forward.w2":
			s.Role, s.InDim, s.OutDim = RoleMLPW2, cfg.IntermediateSize, cfg.EmbDim
		case "feed_forward.w3":
			s.Role, s.InDim, s.OutDim = RoleMLPW3, cfg.EmbDim, cfg.IntermediateSize
		default:
			return LinearSpec{}, false
		}
	}
	return s, s.Role != RoleUnknown && s.InDim > 0 && s.OutDim > 0
}

func RequiredLinearSpecs(cfg Config) ([]LinearSpec, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	prefixes := []string{"input_proj", "llm_cond_proj", "t_embedding.mlp_in", "t_embedding.mlp_out", "adaln_proj", "final_layer.adaln_modulation", "final_layer.linear"}
	for i := 0; i < cfg.NumLayers; i++ {
		prefixes = append(prefixes,
			fmt.Sprintf("layers.%d.adaln_modulation", i),
			fmt.Sprintf("layers.%d.attention.qkv", i),
			fmt.Sprintf("layers.%d.attention.o", i),
			fmt.Sprintf("layers.%d.feed_forward.w1", i),
			fmt.Sprintf("layers.%d.feed_forward.w2", i),
			fmt.Sprintf("layers.%d.feed_forward.w3", i),
		)
	}
	out := make([]LinearSpec, 0, len(prefixes))
	for _, prefix := range prefixes {
		spec, ok := LinearSpecForPrefix(cfg, prefix)
		if !ok {
			return nil, fmt.Errorf("invalid Ideogram4 linear prefix %q", prefix)
		}
		out = append(out, spec)
	}
	return out, nil
}

type LinearCoverage struct {
	Required      int      `json:"required"`
	Present       int      `json:"present"`
	Scaled        int      `json:"scaled"`
	Missing       []string `json:"missing,omitempty"`
	MissingScales []string `json:"missing_scales,omitempty"`
}

func ValidateLinearCoverage(cfg Config, names []string) (LinearCoverage, error) {
	specs, err := RequiredLinearSpecs(cfg)
	if err != nil {
		return LinearCoverage{}, err
	}
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}
	out := LinearCoverage{Required: len(specs)}
	for _, spec := range specs {
		if set[spec.Weight] {
			out.Present++
		} else {
			out.Missing = append(out.Missing, spec.Weight)
		}
		if set[spec.WeightScale] {
			out.Scaled++
		} else {
			out.MissingScales = append(out.MissingScales, spec.WeightScale)
		}
	}
	return out, nil
}

// FP8LinearLoader is the backend boundary for materializing an Ideogram4 FP8
// linear matrix from packed `.weight` (float8) plus `.weight_scale` tensors.
// Implementations live in the owning backend package (FP8 dequant/GEMV), not
// here; this contract only pins the role/shape expectations the runtime needs.
type FP8LinearLoader interface {
	// LoadLinear returns a backend-owned handle for the given linear spec,
	// validating that the provided weight and scale tensors match spec dims.
	LoadLinear(spec LinearSpec) (FP8LinearWeight, error)
}

// FP8LinearWeight is an opaque backend handle to a loaded FP8 linear matrix.
// The runtime only needs its declared shape to wire DiT forward passes; the
// backend owns dequant and matmul execution.
type FP8LinearWeight interface {
	Role() LinearRole
	InDim() int
	OutDim() int
}

func splitLayerPrefix(prefix string) (int, string, bool) {
	parts := strings.Split(prefix, ".")
	if len(parts) < 3 || parts[0] != "layers" {
		return 0, "", false
	}
	idx, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, "", false
	}
	return idx, strings.Join(parts[2:], "."), true
}
