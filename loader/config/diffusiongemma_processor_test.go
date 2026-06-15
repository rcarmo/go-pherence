package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadDiffusionGemmaProcessorConfigPromotesNestedImageProcessorFields(t *testing.T) {
	dir := t.TempDir()
	body := `{
  "processor_class": "Gemma4Processor",
  "image_processor": {
    "do_convert_rgb": true,
    "do_normalize": false,
    "do_rescale": true,
    "do_resize": true,
    "image_mean": [0.0, 0.0, 0.0],
    "image_processor_type": "Gemma4ImageProcessor",
    "image_seq_length": 280,
    "image_std": [1.0, 1.0, 1.0],
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "rescale_factor": 0.00392156862745098
  },
  "video_processor": {
    "video_processor_type": "Gemma4VideoProcessor",
    "patch_size": 16
  }
}`
	if err := os.WriteFile(filepath.Join(dir, "processor_config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err := ReadDiffusionGemmaProcessorConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("processor_config.json not found")
	}
	if cfg.ProcessorClass != "Gemma4Processor" || cfg.ImageProcessorType != "Gemma4ImageProcessor" || cfg.VideoProcessorType != "Gemma4VideoProcessor" {
		t.Fatalf("processor fields=%+v", cfg)
	}
	if cfg.ImageSeqLength != 280 || cfg.PatchSize != 16 || cfg.PoolingKernelSize != 3 {
		t.Fatalf("image fields image_seq=%d patch=%d pooling=%d", cfg.ImageSeqLength, cfg.PatchSize, cfg.PoolingKernelSize)
	}
	if !cfg.DoConvertRGB || !cfg.DoResize || !cfg.DoRescale || cfg.DoNormalize {
		t.Fatalf("image processor flags=%+v", cfg)
	}
	if len(cfg.ImageMean) != 3 || len(cfg.ImageStd) != 3 || cfg.ImageStd[0] != 1 || cfg.RescaleFactor == 0 {
		t.Fatalf("image processor numeric fields=%+v", cfg)
	}
}
