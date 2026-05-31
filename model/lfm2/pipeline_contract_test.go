package lfm2

import (
	"errors"
	"testing"
)

func TestPipelineExecutionContract(t *testing.T) {
	cfg, plan := testGenerationContractPlanConfig(t)
	contract, err := NewPipelineExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.Generation.MaxSequence != 7 || contract.Embedding.PromptTokens != 3 || contract.Conv.HiddenFloats != contract.Attention.HiddenFloats || contract.Attention.HiddenFloats != contract.MoE.HiddenFloats {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestPipelineExecutionContractRejectsMismatch(t *testing.T) {
	cfg, plan := testGenerationContractPlanConfig(t)
	contract, err := NewPipelineExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	contract.Embedding.HiddenSize = contract.Conv.HiddenSize + 1
	contract.Embedding.OutputFloats = contract.Embedding.PromptTokens * contract.Embedding.HiddenSize
	contract.Embedding.Embedding.HiddenSize = contract.Embedding.HiddenSize
	contract.Embedding.Embedding.EmbeddingFloats = contract.Embedding.Embedding.VocabSize * contract.Embedding.HiddenSize
	if !contract.Embedding.Embedding.TieWordEmbeddings {
		contract.Embedding.Embedding.LMHeadFloats = contract.Embedding.Embedding.EmbeddingFloats
	}
	contract.Embedding.Embedding.TotalUntiedFloats = contract.Embedding.Embedding.EmbeddingFloats + contract.Embedding.Embedding.LMHeadFloats
	if err := contract.Validate(); !errors.Is(err, ErrPipelineContractMismatch) {
		t.Fatalf("expected mismatch, got %v", err)
	}
}

func testGenerationContractPlanConfig(t *testing.T) (Config, RuntimeRequestPlan) {
	t.Helper()
	cfg, err := ParseConfig([]byte(`{"model_type":"lfm2_moe","vocab_size":128000,"hidden_size":2048,"num_hidden_layers":3,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv","conv","full_attention"],"num_dense_layers":1,"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Tokens: []uint32{1, 2, 3}, MaxNewTokens: 4, BytesPerFloat: 2})
	if err != nil {
		t.Fatal(err)
	}
	return cfg, plan
}
