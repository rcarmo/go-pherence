package config

import "path/filepath"

type DiffusionGemmaTokenizerConfig struct {
	Backend        string `json:"backend"`
	BOSToken       string `json:"bos_token"`
	EOSToken       string `json:"eos_token"`
	PadToken       string `json:"pad_token"`
	MaskToken      string `json:"mask_token"`
	BOIToken       string `json:"boi_token"`
	EOIToken       string `json:"eoi_token"`
	ImageToken     string `json:"image_token"`
	BOTToken       string `json:"bot_token"`
	SOTToken       string `json:"sot_token"`
	EOTToken       string `json:"eot_token"`
	BOCToken       string `json:"boc_token"`
	SOCToken       string `json:"soc_token"`
	EOCToken       string `json:"eoc_token"`
	ThinkToken     string `json:"think_token"`
	TokenizerClass string `json:"tokenizer_class"`
	ChatTemplate   string `json:"chat_template,omitempty"`
}

type DiffusionGemmaProcessorConfig struct {
	ProcessorClass     string `json:"processor_class"`
	ImageProcessorType string `json:"image_processor_type"`
	VideoProcessorType string `json:"video_processor_type"`
	AudioSeqLength     int    `json:"audio_seq_length"`
	AudioMsPerToken    int    `json:"audio_ms_per_token"`
	ImageSeqLength     int    `json:"image_seq_length"`
	PatchSize          int    `json:"patch_size"`
	TokenBudgetOptions []int  `json:"token_budget_options"`
}

func ReadDiffusionGemmaTokenizerConfig(dir string) (DiffusionGemmaTokenizerConfig, bool, error) {
	var cfg DiffusionGemmaTokenizerConfig
	ok, err := ReadOptionalJSON(filepath.Join(dir, "tokenizer_config.json"), &cfg)
	if err != nil || !ok {
		return cfg, ok, err
	}
	return cfg, true, nil
}

func ReadDiffusionGemmaProcessorConfig(dir string) (DiffusionGemmaProcessorConfig, bool, error) {
	var cfg DiffusionGemmaProcessorConfig
	ok, err := ReadOptionalJSON(filepath.Join(dir, "processor_config.json"), &cfg)
	return cfg, ok, err
}
