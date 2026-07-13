package mosstranscribe

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadUpstreamConfigContract(t *testing.T) {
	cfg, err := LoadConfig(filepath.Join("testdata", "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.AudioTokenID != 151671 || cfg.AdaptorInput != 4096 || cfg.Text.MaxPositions != 131072 {
		t.Fatalf("unexpected upstream config: %+v", cfg)
	}
	whisper := cfg.WhisperConfig()
	if whisper.NumMelBins != 80 || whisper.MaxLength != 3000 || whisper.EncoderLayers != 24 || whisper.EncoderDModel != 1024 || whisper.EncoderHeads != 16 || whisper.EncoderFFNDim != 4096 || whisper.HeadDim != 64 {
		t.Fatalf("unexpected Whisper mapping: %+v", whisper)
	}
}

func TestConfigRejectsUnsupportedGraphChanges(t *testing.T) {
	base, err := LoadConfig(filepath.Join("testdata", "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Config)
		want string
	}{
		{"merge", func(c *Config) { c.AudioMergeSize = 2 }, "merge=2"},
		{"adaptor", func(c *Config) { c.AdaptorInput = 2048 }, "adaptor input=2048"},
		{"audio", func(c *Config) { c.Audio.NumLayers = 23 }, "Whisper encoder"},
		{"decoder", func(c *Config) { c.Text.NumKVHeads = 4 }, "Qwen3 decoder"},
		{"context", func(c *Config) { c.Text.MaxPositions = 40960 }, "Qwen3 decoder"},
		{"layer type", func(c *Config) { c.Text.LayerTypes[3] = "sliding_attention" }, "layer 3 type"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.Text.LayerTypes = append([]string(nil), base.Text.LayerTypes...)
			tc.edit(&cfg)
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() error=%v, want substring %q", err, tc.want)
			}
		})
	}
}
