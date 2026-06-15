package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadDiffusionGemmaGenerationConfigPreservesMissingSentinelsAndExplicitZero(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "generation_config.json")
	if err := os.WriteFile(path, []byte(`{"max_new_tokens":16,"sampler_config":{}}`), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err := ReadDiffusionGemmaGenerationConfig(dir)
	if err != nil || !ok {
		t.Fatalf("ReadDiffusionGemmaGenerationConfig missing controls: ok=%v err=%v", ok, err)
	}
	if cfg.StabilityThreshold != -1 || cfg.ConfidenceThreshold != -1 || cfg.SamplerConfig.EntropyBound != -1 {
		t.Fatalf("missing controls not sentinel: stability=%d confidence=%g entropy=%g", cfg.StabilityThreshold, cfg.ConfidenceThreshold, cfg.SamplerConfig.EntropyBound)
	}

	if err := os.WriteFile(path, []byte(`{"stability_threshold":0,"confidence_threshold":0,"sampler_config":{"entropy_bound":0}}`), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err = ReadDiffusionGemmaGenerationConfig(dir)
	if err != nil || !ok {
		t.Fatalf("ReadDiffusionGemmaGenerationConfig explicit zero: ok=%v err=%v", ok, err)
	}
	if cfg.StabilityThreshold != 0 || cfg.ConfidenceThreshold != 0 || cfg.SamplerConfig.EntropyBound != 0 {
		t.Fatalf("explicit zero controls not preserved: stability=%d confidence=%g entropy=%g", cfg.StabilityThreshold, cfg.ConfidenceThreshold, cfg.SamplerConfig.EntropyBound)
	}
}
