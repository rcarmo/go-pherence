package pockettts

import (
	"os"
	"testing"
)

func TestParseReleasedEnglishConfig(t *testing.T) {
	data, err := os.ReadFile("testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.FlowLM.Transformer.DModel != 1024 || cfg.FlowLM.Transformer.NumLayers != 6 || cfg.FlowLM.Transformer.NumHeads != 16 || cfg.FlowLM.Flow.Dim != 512 || cfg.FlowLM.Flow.Depth != 6 {
		t.Fatalf("FlowLM=%+v", cfg.FlowLM)
	}
	if cfg.Mimi.SampleRate != SampleRate || cfg.Mimi.FrameRate != 12.5 || cfg.Mimi.InnerDim != 32 || cfg.Mimi.OuterDim != 512 || cfg.Mimi.SEANet.Dimension != 512 || cfg.Mimi.Transformer.NumLayers != 2 {
		t.Fatalf("Mimi=%+v", cfg.Mimi)
	}
	if got, err := cfg.SamplesForFrames(3); err != nil || got != 5760 {
		t.Fatalf("samples=%d err=%v", got, err)
	}
}

func TestPocketConfigRejectsMalformed(t *testing.T) {
	data, err := os.ReadFile("testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.FlowLM.Transformer.DModel = 1023 },
		func(c *Config) { c.Mimi.SampleRate = 16000 },
		func(c *Config) { c.Mimi.FrameRate = 12 },
		func(c *Config) { c.Mimi.SEANet.Ratios[0] = 0 },
		func(c *Config) { c.Mimi.Quantizer.Dimension = 31 },
	} {
		bad := cfg
		bad.Mimi.SEANet.Ratios = append([]int(nil), cfg.Mimi.SEANet.Ratios...)
		mutate(&bad)
		if err := bad.Validate(); err == nil {
			t.Fatalf("accepted malformed config %+v", bad)
		}
	}
	if _, err := cfg.SamplesForFrames(0); err == nil {
		t.Fatal("accepted zero frames")
	}
}
