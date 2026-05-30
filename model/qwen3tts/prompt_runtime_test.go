package qwen3tts

import "testing"

func TestPromptRuntimeLayoutAppliesPrefillTalkerInputContracts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"text_hidden_size":2048,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewPromptRuntimeLayout(cfg, PromptIDs{Text: text, Codec: codec})
	if err != nil {
		t.Fatal(err)
	}
	if layout.Prefill.TextTokens != 10 || layout.TalkerInput.TextTokens != 10 || layout.Prefill.EmbeddingFloats != layout.TalkerInput.FusedInputFloats {
		t.Fatalf("layout=%+v", layout)
	}
	layout.TalkerInput.CodecTokens++
	if err := layout.Validate(); err == nil {
		t.Fatal("expected token count mismatch")
	}
	layout.TalkerInput.CodecTokens = layout.Prefill.CodecTokens
	layout.TalkerInput.FusedInputFloats++
	if err := layout.Validate(); err == nil {
		t.Fatal("expected fused input mismatch")
	}
}
