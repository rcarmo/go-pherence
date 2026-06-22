package config

import "path/filepath"

type MiniCPMVGenerationConfig struct {
	MaxNewTokens      int      `json:"max_new_tokens"`
	MaxLength         int      `json:"max_length"`
	DoSample          bool     `json:"do_sample"`
	Temperature       float64  `json:"temperature"`
	TopP              float64  `json:"top_p"`
	TopK              int      `json:"top_k"`
	RepetitionPenalty float64  `json:"repetition_penalty"`
	BOSTokenID        int      `json:"bos_token_id"`
	EOSTokenID        any      `json:"eos_token_id"`
	PadTokenID        int      `json:"pad_token_id"`
	StopStrings       []string `json:"stop_strings"`
}

func ReadMiniCPMVGenerationConfig(dir string) (MiniCPMVGenerationConfig, bool, error) {
	var cfg MiniCPMVGenerationConfig
	ok, err := ReadOptionalJSON(filepath.Join(dir, "generation_config.json"), &cfg)
	if err != nil || !ok {
		return cfg, ok, err
	}
	return cfg, true, nil
}
