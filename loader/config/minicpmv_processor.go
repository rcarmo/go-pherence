package config

import "path/filepath"

// MiniCPMVProcessorConfig captures the image processor sidecars used by
// OpenBMB/Hugging Face MiniCPM-V checkpoints. Repos vary between
// preprocessor_config.json and processor_config.json, and between top-level and
// nested image_processor fields, so NormalizeMiniCPMVProcessorConfig promotes
// common fields into one surface.
type MiniCPMVProcessorConfig struct {
	ProcessorClass     string    `json:"processor_class"`
	ImageProcessorType string    `json:"image_processor_type"`
	Size               any       `json:"size"`
	CropSize           any       `json:"crop_size"`
	DoConvertRGB       bool      `json:"do_convert_rgb"`
	DoNormalize        bool      `json:"do_normalize"`
	DoRescale          bool      `json:"do_rescale"`
	DoResize           bool      `json:"do_resize"`
	ImageMean          []float32 `json:"image_mean"`
	ImageStd           []float32 `json:"image_std"`
	RescaleFactor      float32   `json:"rescale_factor"`
	PatchSize          int       `json:"patch_size"`
	ImageSeqLength     int       `json:"image_seq_length"`
	ImageProcessor     *struct {
		ImageProcessorType string    `json:"image_processor_type"`
		Size               any       `json:"size"`
		CropSize           any       `json:"crop_size"`
		DoConvertRGB       bool      `json:"do_convert_rgb"`
		DoNormalize        bool      `json:"do_normalize"`
		DoRescale          bool      `json:"do_rescale"`
		DoResize           bool      `json:"do_resize"`
		ImageMean          []float32 `json:"image_mean"`
		ImageStd           []float32 `json:"image_std"`
		RescaleFactor      float32   `json:"rescale_factor"`
		PatchSize          int       `json:"patch_size"`
		ImageSeqLength     int       `json:"image_seq_length"`
	} `json:"image_processor"`
	NormalizedSize int `json:"normalized_size"`
}

func ReadMiniCPMVProcessorConfig(dir string) (MiniCPMVProcessorConfig, bool, error) {
	var cfg MiniCPMVProcessorConfig
	for _, name := range []string{"preprocessor_config.json", "processor_config.json"} {
		ok, err := ReadOptionalJSON(filepath.Join(dir, name), &cfg)
		if err != nil {
			return cfg, false, err
		}
		if ok {
			return NormalizeMiniCPMVProcessorConfig(cfg), true, nil
		}
	}
	return cfg, false, nil
}

func NormalizeMiniCPMVProcessorConfig(cfg MiniCPMVProcessorConfig) MiniCPMVProcessorConfig {
	if cfg.ImageProcessor != nil {
		ip := cfg.ImageProcessor
		if cfg.ImageProcessorType == "" {
			cfg.ImageProcessorType = ip.ImageProcessorType
		}
		if cfg.Size == nil {
			cfg.Size = ip.Size
		}
		if cfg.CropSize == nil {
			cfg.CropSize = ip.CropSize
		}
		cfg.DoConvertRGB = ip.DoConvertRGB
		cfg.DoNormalize = ip.DoNormalize
		cfg.DoRescale = ip.DoRescale
		cfg.DoResize = ip.DoResize
		if len(cfg.ImageMean) == 0 {
			cfg.ImageMean = append([]float32(nil), ip.ImageMean...)
		}
		if len(cfg.ImageStd) == 0 {
			cfg.ImageStd = append([]float32(nil), ip.ImageStd...)
		}
		if cfg.RescaleFactor == 0 {
			cfg.RescaleFactor = ip.RescaleFactor
		}
		if cfg.PatchSize == 0 {
			cfg.PatchSize = ip.PatchSize
		}
		if cfg.ImageSeqLength == 0 {
			cfg.ImageSeqLength = ip.ImageSeqLength
		}
	}
	cfg.NormalizedSize = firstPositive(extractSquareSize(cfg.Size), extractSquareSize(cfg.CropSize))
	return cfg
}

func extractSquareSize(v any) int {
	switch x := v.(type) {
	case nil:
		return 0
	case float64:
		if x > 0 {
			return int(x)
		}
	case map[string]any:
		return firstPositive(intFromAny(x["shortest_edge"]), samePositive(intFromAny(x["height"]), intFromAny(x["width"])), intFromAny(x["longest_edge"]))
	}
	return 0
}

func intFromAny(v any) int {
	switch x := v.(type) {
	case float64:
		if x > 0 {
			return int(x)
		}
	case int:
		if x > 0 {
			return x
		}
	}
	return 0
}

func samePositive(a, b int) int {
	if a > 0 && a == b {
		return a
	}
	return 0
}
