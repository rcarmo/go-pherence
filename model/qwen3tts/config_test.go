package qwen3tts

import "testing"

func TestParseConfigCustomVoice(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"tts_model_type":"custom_voice",
		"tts_model_size":"0b6",
		"talker_config":{
			"hidden_size":1024,
			"intermediate_size":3072,
			"num_hidden_layers":28,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"head_dim":64,
			"vocab_size":3072,
			"text_vocab_size":151936,
			"text_hidden_size":2048,
			"rms_norm_eps":0.000001,
			"rope_theta":1000000,
			"max_position_embeddings":32768,
			"rope_scaling":{"mrope_section":[24,20,20]},
			"code_predictor_config":{
				"hidden_size":1024,
				"intermediate_size":3072,
				"num_hidden_layers":5,
				"num_attention_heads":16,
				"num_key_value_heads":8,
				"head_dim":64,
				"vocab_size":2048,
				"num_code_groups":16
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != CustomVoice || cfg.ModelSize != "0b6" {
		t.Fatalf("variant mismatch: %+v", cfg)
	}
	if got := cfg.Label(); got != "0.6B CustomVoice" {
		t.Fatalf("label=%q", got)
	}
	if !cfg.HasMRoPESection || cfg.MRoPESection != [3]int{24, 20, 20} {
		t.Fatalf("mrope=%v %v", cfg.HasMRoPESection, cfg.MRoPESection)
	}
	if cfg.CPNumCodeGroups != 16 {
		t.Fatalf("code groups=%d", cfg.CPNumCodeGroups)
	}
}

func TestParseConfigDerivesMissingHeadDim(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TalkerHeadDim != 64 || cfg.CPHeadDim != 64 {
		t.Fatalf("head dims talker=%d cp=%d", cfg.TalkerHeadDim, cfg.CPHeadDim)
	}
}

func TestParseConfigRejectsBadDims(t *testing.T) {
	_, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1025,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64}}}`))
	if err == nil {
		t.Fatal("expected invalid head dims")
	}
}

func TestTokens(t *testing.T) {
	if id, _ := English.TokenID(); id != 2050 {
		t.Fatalf("english=%d", id)
	}
	if id, _ := Ryan.TokenID(); id != 3061 {
		t.Fatalf("ryan=%d", id)
	}
	lang, err := Ryan.NativeLanguage()
	if err != nil || lang != English {
		t.Fatalf("native=%s err=%v", lang, err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	wantText := []uint32{IMStart, Assistant, Newline, TTSPad, TTSPad, TTSPad, TTSPad, TTSPad, TTSBOS, 123}
	wantCodec := []uint32{CodecThink, CodecThinkBOS, 2050, CodecThinkEOS, 3061, CodecPad, CodecBOS}
	if CustomVoiceFirstTextIndex >= len(text) || text[CustomVoiceFirstTextIndex] != 123 {
		t.Fatalf("first text index=%d text=%v", CustomVoiceFirstTextIndex, text)
	}
	if !eq(text, wantText) {
		t.Fatalf("text=%v", text)
	}
	if !eq(codec, wantCodec) {
		t.Fatalf("codec=%v", codec)
	}
}

func TestTensorCoverage(t *testing.T) {
	cov := InspectTensorNames([]string{
		"talker.model.layers.0.self_attn.q_proj.weight",
		"model.codec_embedding.0.weight",
		"speech_tokenizer.decoder.up.weight",
		"speaker_encoder.blocks.0.weight",
		"misc.weight",
	})
	if cov.Talker != 1 || cov.CodePredictor != 1 || cov.SpeechTokenizer != 1 || cov.SpeakerEncoder != 1 || cov.Other != 1 {
		t.Fatalf("coverage=%+v", cov)
	}
	if !cov.Readiness.Ready {
		t.Fatalf("readiness=%+v", cov.Readiness)
	}
	if !cov.Readiness.PresentOptional["speaker_encoder"] {
		t.Fatalf("optional readiness=%+v", cov.Readiness)
	}
}

func TestTensorReadinessReportsMissingGroups(t *testing.T) {
	ready := InspectTensorReadiness([]string{"talker.model.layers.0.weight"})
	if ready.Ready {
		t.Fatal("expected incomplete tensor readiness")
	}
	want := []string{"code_predictor", "speech_tokenizer"}
	if len(ready.MissingRequired) != len(want) {
		t.Fatalf("missing=%v", ready.MissingRequired)
	}
	for i := range want {
		if ready.MissingRequired[i] != want[i] {
			t.Fatalf("missing=%v", ready.MissingRequired)
		}
	}
}

func eq(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
