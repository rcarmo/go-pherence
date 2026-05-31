package qwen3tts

import (
	"errors"
	"testing"
)

func testPipelineContractConfig(t *testing.T) (ParsedConfig, RuntimeRequestPlan) {
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

func TestPipelineExecutionContract(t *testing.T) {
	cfg, plan := testPipelineContractConfig(t)
	contract, err := NewPipelineExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.Talker.MaxTokens != 2 || contract.CodePredictor.MaxAcousticCodes != 30 || contract.Decoder12Hz.MaxSamples != 4000 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidateStageOutputs([]uint32{0, 1}, make([]uint32, 30), make([]float32, 4000)); err != nil {
		t.Fatal(err)
	}
}

func TestPipelineExecutionContractRejectsMismatch(t *testing.T) {
	cfg, plan := testPipelineContractConfig(t)
	contract, err := NewPipelineExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	contract.CodePredictor.MaxFrames = contract.Decoder12Hz.MaxFrames + 1
	contract.CodePredictor.MaxAcousticCodes = contract.CodePredictor.MaxFrames * contract.CodePredictor.CodesPerFrame
	contract.CodePredictor.Plan.MaxFrames = contract.CodePredictor.MaxFrames
	contract.CodePredictor.Plan.MaxCodes = contract.CodePredictor.MaxAcousticCodes
	contract.CodePredictor.Plan.MaxSamples = contract.CodePredictor.Plan.MaxFrames * contract.CodePredictor.Plan.Waveform.SamplesPerFrame
	if err := contract.Validate(); !errors.Is(err, ErrPipelineContractMismatch) {
		t.Fatalf("expected mismatch, got %v", err)
	}
}

func TestPipelineExecutionContractRejectsMalformedStageOutputs(t *testing.T) {
	cfg, plan := testPipelineContractConfig(t)
	contract, err := NewPipelineExecutionContract(cfg, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateStageOutputs([]uint32{0, 1, 2}, make([]uint32, 30), make([]float32, 4000)); err == nil {
		t.Fatal("expected semantic output error")
	}
	if err := contract.ValidateStageOutputs([]uint32{0, 1}, []uint32{1, 2, 3}, make([]float32, 4000)); err == nil {
		t.Fatal("expected acoustic output error")
	}
	if err := contract.ValidateStageOutputs([]uint32{0, 1}, make([]uint32, 30), make([]float32, 2001)); err == nil {
		t.Fatal("expected waveform output error")
	}
}
