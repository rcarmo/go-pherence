package qwen3tts

import "testing"

func TestTalkerExecutionContract(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 4})
	if err != nil {
		t.Fatal(err)
	}
	contract, err := NewTalkerExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.MaxTokens != 4 || contract.SemanticLayout.Group != 0 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateOutput([]uint32{0, 1, 2, 3}); err != nil {
		t.Fatal(err)
	}
}

func TestTalkerExecutionContractRejectsMalformedOutput(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 2})
	if err != nil {
		t.Fatal(err)
	}
	contract, err := NewTalkerExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput(nil); err == nil {
		t.Fatal("expected empty output error")
	}
	if err := contract.ValidateOutput([]uint32{0, 1, 2}); err == nil {
		t.Fatal("expected max token error")
	}
	if err := contract.ValidateOutput([]uint32{uint32(contract.SemanticLayout.VocabSize)}); err == nil {
		t.Fatal("expected vocab error")
	}
}
