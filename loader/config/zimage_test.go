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
		t.Fatalf("Z-Image runtime must remain unready until DiT/VAE generation is implemented")
	}
	if cfg.VAE.LatentChannels != 16 || cfg.VAE.ScalingFactor != 0.3611 || cfg.VAE.ShiftFactor != 0.1159 || len(cfg.VAE.BlockOutChannels) != 4 {
		t.Fatalf("unexpected pinned VAE boundary: %+v", cfg.VAE)
	}
	for i, want := range []int{128, 256, 512, 512} {
		if cfg.VAE.BlockOutChannels[i] != want {
			t.Fatalf("VAE block %d=%d want=%d", i, cfg.VAE.BlockOutChannels[i], want)
		}
	}
}
