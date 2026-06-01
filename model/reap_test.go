package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadREAPConfig(t *testing.T) {
	dir := t.TempDir()
	data := `{"enabled":true,"prune_ratio":0.2,"active_experts":[0,2],"layers":{"3":[1]}}`
	if err := os.WriteFile(filepath.Join(dir, "reap_config.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadREAPConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil || !cfg.Allows(0, 2) || cfg.Allows(0, 1) || !cfg.Allows(3, 1) || cfg.Allows(3, 2) {
		t.Fatalf("unexpected REAP mask behavior: %+v", cfg)
	}
}

func TestInferREAPConfigFromName(t *testing.T) {
	cfg := InferREAPConfigFromName("Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf")
	if cfg == nil || cfg.PruneRatio != 0.20 || !cfg.Enabled {
		t.Fatalf("unexpected inferred cfg: %+v", cfg)
	}
	if got := InferREAPConfigFromName("dense.gguf"); got != nil {
		t.Fatalf("unexpected cfg: %+v", got)
	}
}

func TestLoadREAPConfigRejectsInvalidPruneRatio(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "reap_config.json"), []byte(`{"prune_ratio":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadREAPConfig(dir); err == nil {
		t.Fatal("expected invalid prune_ratio error")
	}
}
