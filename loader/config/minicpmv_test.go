package config

import "testing"

func TestMiniCPMVConfigOmniLMMSummary(t *testing.T) {
	useStartEnd := true
	cfg := MiniCPMVConfig{
		Architectures:     []string{"OmniLMMForCausalLM"},
		ModelType:         "omnilmm",
		HiddenSize:        4096,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		IntermediateSize:  11008,
		VocabSize:         32064,
		MMVisionTower:     "eva02_enormous_patch14_clip_224.laion2b_plus",
		NumQuery:          64,
		ImageSize:         448,
		PatchSize:         14,
		UseImageStartEnd:  &useStartEnd,
	}
	if err := ValidateMiniCPMVConfig(cfg); err != nil {
		t.Fatalf("ValidateMiniCPMVConfig: %v", err)
	}
	s := cfg.MiniCPMVSummary()
	if s.ResamplerGrid != 8 || s.ResamplerHeads != 32 || !s.UseImageStartEnd {
		t.Fatalf("unexpected resampler/start-end summary: %+v", s)
	}
	if s.HeadDim != 128 || s.KVHeads != 8 {
		t.Fatalf("unexpected text head dims: %+v", s)
	}
}

func TestMiniCPMVConfigNestedSummary(t *testing.T) {
	cfg := MiniCPMVConfig{
		Architectures:   []string{"MiniCPMVForCausalLM"},
		ModelType:       "minicpmv",
		TextConfig:      &MiniCPMVTextConfig{ModelType: "qwen2", HiddenSize: 3584, NumHiddenLayers: 28, NumAttentionHeads: 28, NumKeyValueHeads: 4, IntermediateSize: 18944, VocabSize: 151666},
		VisionConfig:    &MiniCPMVVisionConfig{ModelType: "siglip_vision_model", HiddenSize: 1152, NumHiddenLayers: 27, NumAttentionHeads: 16, ImageSize: 448, PatchSize: 14},
		ResamplerConfig: &MiniCPMVResamplerConfig{NumQuery: 64, NumHeads: 28, KVDim: 1152},
	}
	if err := ValidateMiniCPMVConfig(cfg); err != nil {
		t.Fatalf("ValidateMiniCPMVConfig nested: %v", err)
	}
	s := cfg.MiniCPMVSummary()
	if s.TextModelType != "qwen2" || s.VisionModelType != "siglip_vision_model" {
		t.Fatalf("missing nested model types: %+v", s)
	}
	if s.HiddenSize != 3584 || s.VisionHiddenSize != 1152 || s.ImageSize != 448 || s.PatchSize != 14 {
		t.Fatalf("unexpected nested dims: %+v", s)
	}
}

func TestMiniCPMOConfigNestedSummary(t *testing.T) {
	cfg := MiniCPMVConfig{
		Architectures:   []string{"MiniCPMOForCausalLM"},
		ModelType:       "minicpm-o",
		TextConfig:      &MiniCPMVTextConfig{ModelType: "qwen2", HiddenSize: 3584, NumHiddenLayers: 28, NumAttentionHeads: 28, NumKeyValueHeads: 4, IntermediateSize: 18944, VocabSize: 151666},
		VisionConfig:    &MiniCPMVVisionConfig{ModelType: "siglip_vision_model", HiddenSize: 1152, NumHiddenLayers: 27, NumAttentionHeads: 16, ImageSize: 448, PatchSize: 14},
		AudioConfig:     &MiniCPMOAudioConfig{ModelType: "whisper_encoder", HiddenSize: 1280, NumHiddenLayers: 32, NumAttentionHeads: 20, FeatureSize: 128, NumMelBins: 128, SamplingRate: 16000},
		ResamplerConfig: &MiniCPMVResamplerConfig{NumQuery: 64, NumHeads: 28, KVDim: 1152},
	}
	if err := ValidateMiniCPMVConfig(cfg); err != nil {
		t.Fatalf("ValidateMiniCPMVConfig MiniCPM-O: %v", err)
	}
	s := cfg.MiniCPMVSummary()
	if s.ModelType != "minicpm-o" || s.Architecture != "MiniCPMOForCausalLM" || s.TextModelType != "qwen2" {
		t.Fatalf("bad MiniCPM-O summary: %+v", s)
	}
	if s.AudioModelType != "whisper_encoder" || s.AudioHiddenSize != 1280 || s.AudioMelBins != 128 || s.AudioSamplingRate != 16000 {
		t.Fatalf("bad MiniCPM-O audio summary: %+v", s)
	}
}

func TestMiniCPMVConfigRejectsNonSquareQuery(t *testing.T) {
	cfg := MiniCPMVConfig{Architectures: []string{"MiniCPMVForCausalLM"}, ModelType: "minicpmv", HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 1, VocabSize: 10, NumQuery: 63}
	if err := ValidateMiniCPMVConfig(cfg); err == nil {
		t.Fatalf("expected non-square num_query to fail")
	}
}
