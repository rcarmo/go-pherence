package minicpmv

import "github.com/rcarmo/go-pherence/loader/config"

func ImagePreprocessConfigFromProcessor(proc config.MiniCPMVProcessorConfig, fallbackSize, fallbackPatchSize int) ImagePreprocessConfig {
	cfg := DefaultImagePreprocessConfig(firstPositive(proc.NormalizedSize, fallbackSize), firstPositive(proc.PatchSize, fallbackPatchSize))
	cfg.DoConvertRGB = proc.DoConvertRGB
	cfg.DoResize = proc.DoResize
	cfg.DoRescale = proc.DoRescale
	cfg.DoNormalize = proc.DoNormalize
	if proc.RescaleFactor != 0 {
		cfg.RescaleFactor = proc.RescaleFactor
	}
	if len(proc.ImageMean) >= 3 {
		copy(cfg.ImageMean[:], proc.ImageMean[:3])
	}
	if len(proc.ImageStd) >= 3 {
		copy(cfg.ImageStd[:], proc.ImageStd[:3])
	}
	return cfg
}

func firstPositive(v int, rest ...int) int {
	if v > 0 {
		return v
	}
	for _, x := range rest {
		if x > 0 {
			return x
		}
	}
	return 0
}
