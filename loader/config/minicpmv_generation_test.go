package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadMiniCPMVGenerationConfig(t *testing.T) {
	dir := t.TempDir()
	data := `{
  "max_new_tokens": 256,
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "bos_token_id": 1,
  "eos_token_id": [2, 3],
  "pad_token_id": 0,
  "stop_strings": ["<im_end>"]
}`
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, ok, err := ReadMiniCPMVGenerationConfig(dir)
	if err != nil || !ok {
		t.Fatalf("ReadMiniCPMVGenerationConfig ok=%v err=%v", ok, err)
	}
	if cfg.MaxNewTokens != 256 || !cfg.DoSample || cfg.Temperature != 0.7 || cfg.TopP != 0.9 || cfg.TopK != 50 || len(cfg.StopStrings) != 1 {
		t.Fatalf("bad generation config: %+v", cfg)
	}
}

func TestReadMiniCPMVGenerationConfigMissing(t *testing.T) {
	_, ok, err := ReadMiniCPMVGenerationConfig(t.TempDir())
	if err != nil || ok {
		t.Fatalf("missing generation ok=%v err=%v", ok, err)
	}
}
