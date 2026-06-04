package config

import "testing"

func TestReadZImageTurboConfig(t *testing.T) {
	cfg, err := ReadZImageConfig("../../testdata/zimage")
	if err != nil {
		t.Fatalf("ReadZImageConfig: %v", err)
	}
	s := SummarizeZImageConfig(cfg)
	if s.Pipeline != "ZImagePipeline" || s.Transformer != "ZImageTransformer2DModel" || s.Scheduler != "FlowMatchEulerDiscreteScheduler" {
		t.Fatalf("unexpected components: %+v", s)
	}
	if s.Dim != 3840 || s.Layers != 30 || s.RefinerLayers != 2 || s.Heads != 30 || s.InChannels != 16 || s.CapFeatDim != 2560 {
		t.Fatalf("unexpected transformer summary: %+v", s)
	}
	if s.TextEncoder != "Qwen3Model" || s.Tokenizer != "Qwen2Tokenizer" || s.TextHidden != 2560 || s.TextLayers != 36 || s.VocabSize != 151936 {
		t.Fatalf("unexpected text summary: %+v", s)
	}
	if s.RuntimeReady {
		t.Fatalf("Z-Image runtime should be inspection-only until DiT/VAE generation is implemented")
	}
}
