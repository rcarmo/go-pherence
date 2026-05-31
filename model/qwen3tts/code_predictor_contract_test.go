package qwen3tts

import "testing"

func testCodePredictorContractPlan(t *testing.T) (ParsedConfig, RuntimeRequestPlan) {
	t.Helper()
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
	return cfg, plan
}

func TestCodePredictorExecutionContract(t *testing.T) {
	cfg, plan := testCodePredictorContractPlan(t)
	contract, err := NewCodePredictorExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.MaxFrames != 2 || contract.CodesPerFrame != 15 || contract.MaxAcousticCodes != 30 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateInput([]uint32{0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput(make([]uint32, 30)); err != nil {
		t.Fatal(err)
	}
}

func TestCodePredictorExecutionContractRejectsMalformed(t *testing.T) {
	cfg, plan := testCodePredictorContractPlan(t)
	contract, err := NewCodePredictorExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateInput(nil); err == nil {
		t.Fatal("expected empty semantic input error")
	}
	if err := contract.ValidateInput([]uint32{0, 1, 2}); err == nil {
		t.Fatal("expected semantic max error")
	}
	if err := contract.ValidateInput([]uint32{uint32(contract.SemanticLayout.VocabSize)}); err == nil {
		t.Fatal("expected semantic vocab error")
	}
	if err := contract.ValidateOutput([]uint32{1, 2, 3}); err == nil {
		t.Fatal("expected partial acoustic frame error")
	}
	tooMany := make([]uint32, contract.MaxAcousticCodes+contract.CodesPerFrame)
	if err := contract.ValidateOutput(tooMany); err == nil {
		t.Fatal("expected max acoustic code error")
	}
	bad := make([]uint32, contract.CodesPerFrame)
	bad[0] = uint32(contract.FrameLayout.CodecVocab)
	if err := contract.ValidateOutput(bad); err == nil {
		t.Fatal("expected acoustic vocab error")
	}
}
