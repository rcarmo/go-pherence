package lfm2

import "testing"

func testEmbeddingContractConfig(t *testing.T) (Config, RuntimeRequestPlan) {
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

func TestEmbeddingExecutionContract(t *testing.T) {
	cfg, plan := testEmbeddingContractConfig(t)
	contract, err := NewEmbeddingExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.PromptTokens != 3 || contract.HiddenSize != 2048 || contract.OutputFloats != 6144 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateInput([]uint32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput(make([]float32, contract.OutputFloats)); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddingExecutionContractRejectsMalformed(t *testing.T) {
	cfg, plan := testEmbeddingContractConfig(t)
	contract, err := NewEmbeddingExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateInput([]uint32{1, 2}); err == nil {
		t.Fatal("expected input length error")
	}
	if err := contract.ValidateInput([]uint32{1, 2, uint32(contract.Context.VocabSize)}); err == nil {
		t.Fatal("expected input vocab error")
	}
	if err := contract.ValidateOutput(make([]float32, contract.OutputFloats-1)); err == nil {
		t.Fatal("expected output size error")
	}
}
