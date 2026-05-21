package gemma

import (
	"encoding/json"

	"github.com/rcarmo/go-pherence/model/common"
)

// NormalizeTextConfig extracts Gemma4's nested text_config into the common
// decoder config shape used by the compatibility model loader.
func NormalizeTextConfig(data []byte, fallback common.Config) (common.Config, bool) {
	var raw struct {
		ModelType     string        `json:"model_type"`
		Architectures []string      `json:"architectures"`
		TextConfig    common.Config `json:"text_config"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return fallback, false
	}
	if raw.ModelType != "gemma4" || raw.TextConfig.HiddenSize <= 0 {
		return fallback, false
	}
	cfg := raw.TextConfig
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_text"
	}
	if len(cfg.Architectures) == 0 {
		cfg.Architectures = append([]string(nil), raw.Architectures...)
	}
	return cfg, true
}

// LayerKVHeads returns the K/V head count for a Gemma layer. Gemma4 31B uses
// fewer K/V heads for full-attention layers than for sliding-attention layers.
func LayerKVHeads(cfg common.Config, layerIdx int) int {
	if cfg.NumGlobalKVHeads > 0 && layerIdx >= 0 && layerIdx < len(cfg.LayerTypes) && cfg.LayerTypes[layerIdx] == "full_attention" {
		return cfg.NumGlobalKVHeads
	}
	return cfg.NumKVHeads
}
