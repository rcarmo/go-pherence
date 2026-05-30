package qwen3tts

import "testing"

func TestSpeakerEncoderLayoutPresent(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"tts_model_type":"custom_voice",
		"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64}},
		"speaker_encoder_config":{"enc_dim":1024,"sample_rate":24000}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewSpeakerEncoderLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !layout.Present || layout.EmbeddingDim != 1024 || layout.SampleRateHz != 24000 || layout.EmbeddingFloats != 1024 || layout.ReferenceChannels != 1 {
		t.Fatalf("layout=%+v", layout)
	}
	samples, err := layout.ReferenceSamples(3)
	if err != nil {
		t.Fatal(err)
	}
	if samples != 72000 {
		t.Fatalf("samples=%d", samples)
	}
}

func TestSpeakerEncoderLayoutAbsent(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64}}}`))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewSpeakerEncoderLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Present {
		t.Fatalf("layout=%+v", layout)
	}
	if _, err := layout.ReferenceSamples(1); err == nil {
		t.Fatal("expected absent speaker encoder error")
	}
}

func TestSpeakerEncoderLayoutRejectsMalformed(t *testing.T) {
	bad := SpeakerEncoderLayout{Present: true, EmbeddingDim: 1024, SampleRateHz: 24000, EmbeddingFloats: 1, ReferenceChannels: 1, SamplesPerSecond: 24000}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected embedding float error")
	}
	bad = SpeakerEncoderLayout{Present: true, EmbeddingDim: 1024, SampleRateHz: 24000, EmbeddingFloats: 1024, ReferenceChannels: 2, SamplesPerSecond: 24000}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected samples/second error")
	}
	bad = SpeakerEncoderLayout{Present: false, EmbeddingDim: 1024}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected absent layout error")
	}
	if _, err := (SpeakerEncoderLayout{Present: true, EmbeddingDim: 1024, SampleRateHz: 24000, EmbeddingFloats: 1024, ReferenceChannels: 1, SamplesPerSecond: 24000}).ReferenceSamples(-1); err == nil {
		t.Fatal("expected negative duration error")
	}
}
