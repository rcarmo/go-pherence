package lfm2

import "testing"

func testAttentionContractConfig(t *testing.T) (Config, RuntimeRequestPlan) {
	t.Helper()
	cfg, err := ParseConfig([]byte(`{"model_type":"lfm2_moe","vocab_size":128000,"hidden_size":2048,"num_hidden_layers":3,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv","conv","full_attention"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Tokens: []uint32{1, 2, 3}, MaxNewTokens: 4, BytesPerFloat: 2})
	if err != nil {
		t.Fatal(err)
	}
	return cfg, plan
}

func TestAttentionExecutionContract(t *testing.T) {
	cfg, plan := testAttentionContractConfig(t)
	contract, err := NewAttentionExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.SequenceTokens != 7 || contract.HiddenSize != 2048 || contract.HiddenFloats != 14336 || contract.KVFloats != 7168 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateInput(make([]float32, contract.HiddenFloats), make([]float32, contract.KVFloats)); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput(make([]float32, contract.HiddenFloats), make([]float32, contract.KVFloats)); err != nil {
		t.Fatal(err)
	}
}

func TestAttentionExecutionContractRejectsMalformed(t *testing.T) {
	cfg, plan := testAttentionContractConfig(t)
	contract, err := NewAttentionExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateInput(make([]float32, contract.HiddenFloats-1), make([]float32, contract.KVFloats)); err == nil {
		t.Fatal("expected hidden input size error")
	}
	if err := contract.ValidateInput(make([]float32, contract.HiddenFloats), make([]float32, contract.KVFloats-1)); err == nil {
		t.Fatal("expected KV input size error")
	}
	if err := contract.ValidateOutput(make([]float32, contract.HiddenFloats+1), make([]float32, contract.KVFloats)); err == nil {
		t.Fatal("expected hidden output size error")
	}
}
