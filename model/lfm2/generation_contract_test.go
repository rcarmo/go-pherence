package lfm2

import "testing"

func testGenerationContractPlan(t *testing.T) RuntimeRequestPlan {
	t.Helper()
	cfg, err := ParseConfig([]byte(`{"model_type":"lfm2_moe","vocab_size":128000,"hidden_size":2048,"num_hidden_layers":3,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv","conv","full_attention"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Tokens: []uint32{1, 2, 3}, MaxNewTokens: 4, BytesPerFloat: 2})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func TestGenerationExecutionContract(t *testing.T) {
	plan := testGenerationContractPlan(t)
	contract, err := NewGenerationExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.PromptTokens != 3 || contract.MaxNewTokens != 4 || contract.MaxSequence != 7 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidatePrompt([]uint32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput([]uint32{4, 5}); err != nil {
		t.Fatal(err)
	}
}

func TestGenerationExecutionContractRejectsMalformed(t *testing.T) {
	plan := testGenerationContractPlan(t)
	contract, err := NewGenerationExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidatePrompt([]uint32{1, 2}); err == nil {
		t.Fatal("expected prompt length error")
	}
	if err := contract.ValidatePrompt([]uint32{1, 2, uint32(contract.Context.VocabSize)}); err == nil {
		t.Fatal("expected prompt vocab error")
	}
	if err := contract.ValidateOutput(nil); err == nil {
		t.Fatal("expected empty output error")
	}
	if err := contract.ValidateOutput([]uint32{1, 2, 3, 4, 5}); err == nil {
		t.Fatal("expected output max error")
	}
	if err := contract.ValidateOutput([]uint32{uint32(contract.Context.VocabSize)}); err == nil {
		t.Fatal("expected output vocab error")
	}
}
