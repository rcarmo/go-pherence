package qwen3tts

import "testing"

func TestRuntimePlanAppliesDecoderInputContract(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	decoderPlan, err := plan.DecoderInputLayout.DecoderPlan()
	if err != nil {
		t.Fatal(err)
	}
	if decoderPlan != plan.Decoder12Hz {
		t.Fatalf("decoder contract not applied: input=%+v plan=%+v", decoderPlan, plan.Decoder12Hz)
	}
	if plan.WaveformLayout.FrameRateHz != plan.DecoderInputLayout.FrameRateHz {
		t.Fatalf("waveform frame rate=%d decoder input=%d", plan.WaveformLayout.FrameRateHz, plan.DecoderInputLayout.FrameRateHz)
	}
}

func TestRuntimePlanRejectsDecoderContractMismatch(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	plan.Decoder12Hz.CodeGroups = 14
	if err := plan.Validate(); err == nil {
		t.Fatal("expected decoder input/plan mismatch")
	}
}
