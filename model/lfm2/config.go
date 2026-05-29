// Package lfm2 contains metadata and validation helpers for Liquid AI LFM2
// checkpoints. Runtime inference is intentionally staged after config and
// tensor inventory coverage.
package lfm2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const ModelType = "lfm2_moe"

type RoPEParameters struct {
	Theta float64 `json:"rope_theta"`
	Type  string  `json:"rope_type"`
}

type Config struct {
	ModelType             string         `json:"model_type"`
	Architectures         []string       `json:"architectures"`
	DType                 string         `json:"dtype"`
	VocabSize             int            `json:"vocab_size"`
	HiddenSize            int            `json:"hidden_size"`
	IntermediateSize      int            `json:"intermediate_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	HeadDim               int            `json:"head_dim"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	LayerTypes            []string       `json:"layer_types"`
	NumDenseLayers        int            `json:"num_dense_layers"`
	NumExperts            int            `json:"num_experts"`
	NumExpertsPerTok      int            `json:"num_experts_per_tok"`
	MoEIntermediateSize   int            `json:"moe_intermediate_size"`
	ConvLCache            int            `json:"conv_L_cache"`
	ConvBias              bool           `json:"conv_bias"`
	NormEps               float64        `json:"norm_eps"`
	NormTopKProb          bool           `json:"norm_topk_prob"`
	RoutedScalingFactor   float64        `json:"routed_scaling_factor"`
	UseExpertBias         bool           `json:"use_expert_bias"`
	TieWordEmbeddings     bool           `json:"tie_word_embeddings"`
	RoPE                  RoPEParameters `json:"rope_parameters"`
}

func ParseConfigFile(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	return ParseConfig(data)
}

func ReadModelDir(dir string) (Config, error) {
	return ParseConfigFile(filepath.Join(dir, "config.json"))
}

func ParseConfig(data []byte) (Config, error) {
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return Config{}, err
	}
	if c.HeadDim == 0 && c.NumAttentionHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	return c, c.Validate()
}

func (c Config) Validate() error {
	if c.ModelType != ModelType {
		return fmt.Errorf("unsupported LFM2 model_type %q", c.ModelType)
	}
	if c.HiddenSize <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.HeadDim <= 0 {
		return fmt.Errorf("invalid LFM2 attention dims: hidden=%d heads=%d kv_heads=%d head_dim=%d", c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim)
	}
	if c.HiddenSize != c.NumAttentionHeads*c.HeadDim {
		return fmt.Errorf("invalid LFM2 head dims: hidden=%d heads=%d head_dim=%d", c.HiddenSize, c.NumAttentionHeads, c.HeadDim)
	}
	if c.NumHiddenLayers <= 0 || len(c.LayerTypes) != c.NumHiddenLayers {
		return fmt.Errorf("invalid LFM2 layer pattern: layers=%d layer_types=%d", c.NumHiddenLayers, len(c.LayerTypes))
	}
	if c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts || c.MoEIntermediateSize <= 0 {
		return fmt.Errorf("invalid LFM2 MoE dims: experts=%d active=%d moe_intermediate=%d", c.NumExperts, c.NumExpertsPerTok, c.MoEIntermediateSize)
	}
	if c.ConvLCache <= 0 {
		return fmt.Errorf("invalid LFM2 conv_L_cache %d", c.ConvLCache)
	}
	for i, typ := range c.LayerTypes {
		if typ != "conv" && typ != "full_attention" {
			return fmt.Errorf("unsupported LFM2 layer type at %d: %q", i, typ)
		}
	}
	return nil
}

func (c Config) ConvLayerCount() int          { return count(c.LayerTypes, "conv") }
func (c Config) FullAttentionLayerCount() int { return count(c.LayerTypes, "full_attention") }

func count(xs []string, want string) int {
	n := 0
	for _, x := range xs {
		if x == want {
			n++
		}
	}
	return n
}
