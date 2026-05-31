package qwen3tts

import "testing"

func testDecoderContractPlan(t *testing.T) RuntimeRequestPlan {
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
	return plan
}

func TestDecoder12HzExecutionContract(t *testing.T) {
	plan := testDecoderContractPlan(t)
	contract, err := NewDecoder12HzExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.MaxFrames != 2 || contract.CodesPerFrame != 15 || contract.SamplesPerFrame != 2000 || contract.MaxSamples != 4000 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateInput(make([]uint32, 30)); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput(make([]float32, 4000)); err != nil {
		t.Fatal(err)
	}
}

func TestDecoder12HzExecutionContractRejectsMalformed(t *testing.T) {
	plan := testDecoderContractPlan(t)
	contract, err := NewDecoder12HzExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateInput(nil); err == nil {
		t.Fatal("expected empty acoustic input error")
	}
	if err := contract.ValidateInput([]uint32{1, 2, 3}); err == nil {
		t.Fatal("expected partial acoustic frame error")
	}
	tooManyCodes := make([]uint32, contract.MaxAcousticCodes+contract.CodesPerFrame)
	if err := contract.ValidateInput(tooManyCodes); err == nil {
		t.Fatal("expected max acoustic input error")
	}
	badCodes := make([]uint32, contract.CodesPerFrame)
	badCodes[0] = uint32(contract.DecoderInput.CodecVocab)
	if err := contract.ValidateInput(badCodes); err == nil {
		t.Fatal("expected acoustic vocab error")
	}
	if err := contract.ValidateOutput(nil); err == nil {
		t.Fatal("expected empty sample output error")
	}
	if err := contract.ValidateOutput(make([]float32, contract.SamplesPerFrame+1)); err == nil {
		t.Fatal("expected partial sample frame error")
	}
	if err := contract.ValidateOutput(make([]float32, contract.MaxSamples+contract.SamplesPerFrame)); err == nil {
		t.Fatal("expected max sample output error")
	}
}
