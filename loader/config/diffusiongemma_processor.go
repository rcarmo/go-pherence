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
	ProcessorClass     string    `json:"processor_class"`
	ImageProcessorType string    `json:"image_processor_type"`
	VideoProcessorType string    `json:"video_processor_type"`
	AudioSeqLength     int       `json:"audio_seq_length"`
	AudioMsPerToken    int       `json:"audio_ms_per_token"`
	ImageSeqLength     int       `json:"image_seq_length"`
	PatchSize          int       `json:"patch_size"`
	DoConvertRGB       bool      `json:"do_convert_rgb"`
	DoNormalize        bool      `json:"do_normalize"`
	DoRescale          bool      `json:"do_rescale"`
	DoResize           bool      `json:"do_resize"`
	ImageMean          []float32 `json:"image_mean"`
	ImageStd           []float32 `json:"image_std"`
	PoolingKernelSize  int       `json:"pooling_kernel_size"`
	RescaleFactor      float32   `json:"rescale_factor"`
	TokenBudgetOptions []int     `json:"token_budget_options"`
	ImageProcessor     struct {
		DoConvertRGB       bool      `json:"do_convert_rgb"`
		DoNormalize        bool      `json:"do_normalize"`
		DoRescale          bool      `json:"do_rescale"`
		DoResize           bool      `json:"do_resize"`
		ImageMean          []float32 `json:"image_mean"`
		ImageProcessorType string    `json:"image_processor_type"`
		ImageSeqLength     int       `json:"image_seq_length"`
		ImageStd           []float32 `json:"image_std"`
		PatchSize          int       `json:"patch_size"`
		PoolingKernelSize  int       `json:"pooling_kernel_size"`
		RescaleFactor      float32   `json:"rescale_factor"`
	} `json:"image_processor"`
	VideoProcessor struct {
		VideoProcessorType string `json:"video_processor_type"`
		PatchSize          int    `json:"patch_size"`
	} `json:"video_processor"`
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
	if err != nil || !ok {
		return cfg, ok, err
	}
	if cfg.ImageProcessorType == "" {
		cfg.ImageProcessorType = cfg.ImageProcessor.ImageProcessorType
	}
	if cfg.VideoProcessorType == "" {
		cfg.VideoProcessorType = cfg.VideoProcessor.VideoProcessorType
	}
	if cfg.ImageSeqLength == 0 {
		cfg.ImageSeqLength = cfg.ImageProcessor.ImageSeqLength
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = cfg.ImageProcessor.PatchSize
	}
	cfg.DoConvertRGB = cfg.ImageProcessor.DoConvertRGB
	cfg.DoNormalize = cfg.ImageProcessor.DoNormalize
	cfg.DoRescale = cfg.ImageProcessor.DoRescale
	cfg.DoResize = cfg.ImageProcessor.DoResize
	cfg.ImageMean = append([]float32(nil), cfg.ImageProcessor.ImageMean...)
	cfg.ImageStd = append([]float32(nil), cfg.ImageProcessor.ImageStd...)
	cfg.PoolingKernelSize = cfg.ImageProcessor.PoolingKernelSize
	cfg.RescaleFactor = cfg.ImageProcessor.RescaleFactor
	return cfg, true, nil
}
