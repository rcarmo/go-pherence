package qwen3tts

import "testing"

func TestRuntimePlanAppliesAttentionKVContracts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_hidden_layers":5,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	talkerKV, err := plan.TalkerAttentionLayout.KVFloatsPerToken()
	if err != nil {
		t.Fatal(err)
	}
	cpKV, err := plan.CPAttentionLayout.KVFloatsPerToken()
	if err != nil {
		t.Fatal(err)
	}
	if plan.Talker.KVFloatsPerToken != talkerKV || plan.CodePredictor.KVFloatsPerToken != cpKV {
		t.Fatalf("KV contracts not applied: talker=%d/%d cp=%d/%d", plan.Talker.KVFloatsPerToken, talkerKV, plan.CodePredictor.KVFloatsPerToken, cpKV)
	}
	plan.Talker.KVFloatsPerToken++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected talker KV layout mismatch")
	}
	plan.Talker.KVFloatsPerToken = talkerKV
	plan.CodePredictor.KVFloatsPerToken++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected code predictor KV layout mismatch")
	}
}
