package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadMiniCPMVProcessorConfigPreprocessor(t *testing.T) {
	dir := t.TempDir()
	data := `{
  "image_processor_type": "SiglipImageProcessor",
  "size": {"height": 448, "width": 448},
  "do_resize": true,
  "do_rescale": true,
  "do_normalize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "rescale_factor": 0.00392156862745098
}`
	if err := os.WriteFile(filepath.Join(dir, "preprocessor_config.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err := ReadMiniCPMVProcessorConfig(dir)
	if err != nil || !ok {
		t.Fatalf("ReadMiniCPMVProcessorConfig ok=%v err=%v", ok, err)
	}
	if cfg.ImageProcessorType != "SiglipImageProcessor" || cfg.NormalizedSize != 448 || len(cfg.ImageMean) != 3 || !cfg.DoNormalize {
		t.Fatalf("bad processor config: %+v", cfg)
	}
}

func TestReadMiniCPMVProcessorConfigNestedProcessor(t *testing.T) {
	dir := t.TempDir()
	data := `{
  "processor_class": "MiniCPMVProcessor",
  "image_processor": {
    "image_processor_type": "CLIPImageProcessor",
    "size": {"shortest_edge": 336},
    "do_resize": true,
    "do_rescale": true,
    "do_normalize": true,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "patch_size": 14,
    "image_seq_length": 576
  }
}`
	if err := os.WriteFile(filepath.Join(dir, "processor_config.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err := ReadMiniCPMVProcessorConfig(dir)
	if err != nil || !ok {
		t.Fatalf("ReadMiniCPMVProcessorConfig ok=%v err=%v", ok, err)
	}
	if cfg.ImageProcessorType != "CLIPImageProcessor" || cfg.NormalizedSize != 336 || cfg.PatchSize != 14 || cfg.ImageSeqLength != 576 {
		t.Fatalf("bad nested processor config: %+v", cfg)
	}
}

func TestReadMiniCPMVProcessorConfigMissing(t *testing.T) {
	_, ok, err := ReadMiniCPMVProcessorConfig(t.TempDir())
	if err != nil || ok {
		t.Fatalf("missing processor ok=%v err=%v", ok, err)
	}
}
