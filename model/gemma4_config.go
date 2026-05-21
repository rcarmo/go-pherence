package model

import "encoding/json"

func normalizeGemma4TextConfig(data []byte, fallback LlamaConfig) (LlamaConfig, bool) {
	var raw struct {
		ModelType     string      `json:"model_type"`
		Architectures []string    `json:"architectures"`
		TextConfig    LlamaConfig `json:"text_config"`
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
