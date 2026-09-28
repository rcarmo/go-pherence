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
	if contract.MaxFrames != 2 || contract.CodesPerFrame != 16 || contract.AcousticPerFrame != 15 || contract.SamplesPerFrame != 1920 || contract.MaxDecoderCodes != 32 || contract.MaxAcousticCodes != 30 || contract.MaxSamples != 3840 {
		t.Fatalf("contract=%+v", contract)
	}
	codes, err := contract.JoinInput([]uint32{1, 2}, make([]uint32, 30))
	if err != nil || len(codes) != 32 {
		t.Fatalf("joined codes=%d err=%v", len(codes), err)
	}
	if err := contract.ValidateInput(codes); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutputForFrames(make([]float32, 3840), 2); err != nil {
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
	tooManyCodes := make([]uint32, contract.MaxDecoderCodes+contract.CodesPerFrame)
	if err := contract.ValidateInput(tooManyCodes); err == nil {
		t.Fatal("expected max acoustic input error")
	}
	badCodes := make([]uint32, contract.CodesPerFrame)
	badCodes[1] = uint32(contract.DecoderInput.CodecVocab)
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
	if err := contract.ValidateOutputForFrames(make([]float32, contract.MaxSamples), 1); err == nil {
		t.Fatal("expected frame-exact sample output error")
	}
}
