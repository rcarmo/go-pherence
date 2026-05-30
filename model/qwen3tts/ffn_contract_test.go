package qwen3tts

import "testing"

func TestRuntimePlanAppliesFFNContracts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":5,"num_attention_heads":16,"head_dim":64}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Talker.MatchesFFNLayout(plan.TalkerFFNLayout); err != nil {
		t.Fatal(err)
	}
	if err := plan.CodePredictor.MatchesFFNLayout(plan.CPFFNLayout); err != nil {
		t.Fatal(err)
	}
	plan.Talker.IntermediateSize++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected talker FFN layout mismatch")
	}
	plan.Talker.IntermediateSize = plan.TalkerFFNLayout.IntermediateSize
	plan.CodePredictor.Layers++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected code predictor FFN layout mismatch")
	}
}
