package qwen3tts

import "testing"

func TestRuntimeRequestPlan(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 12})
	if err != nil {
		t.Fatal(err)
	}
	if plan.MaxSamples != 24000 || plan.MaxCodes != 180 || plan.PromptLayout.Prefill.TextTokens != len(text) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestRuntimeRequestPlanFromSeconds(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxSeconds: 1})
	if err != nil {
		t.Fatal(err)
	}
	if plan.MaxFrames != 12 {
		t.Fatalf("frames=%d", plan.MaxFrames)
	}
}

func TestRuntimeRequestPlanRejectsMalformed(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 1}); err == nil {
		t.Fatal("expected conditioning error")
	}
	if _, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}}); err == nil {
		t.Fatal("expected max frames error")
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	plan.MaxCodes++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected code count mismatch")
	}
}
