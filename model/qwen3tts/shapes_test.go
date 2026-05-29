package qwen3tts

import "testing"

func TestRuntimePlanSizing(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"tts_model_type":"custom_voice",
		"tts_model_size":"0b6",
		"talker_config":{
			"hidden_size":1024,
			"intermediate_size":3072,
			"num_hidden_layers":28,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"head_dim":64,
			"vocab_size":3072,
			"code_predictor_config":{
				"hidden_size":1024,
				"intermediate_size":3072,
				"num_hidden_layers":5,
				"num_attention_heads":16,
				"num_key_value_heads":8,
				"head_dim":64,
				"vocab_size":2048,
				"num_code_groups":16
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Talker.KVFloatsPerToken != 28672 {
		t.Fatalf("talker kv floats/token=%d", plan.Talker.KVFloatsPerToken)
	}
	if plan.CodePredictor.KVFloatsPerToken != 5120 {
		t.Fatalf("cp kv floats/token=%d", plan.CodePredictor.KVFloatsPerToken)
	}
	if plan.Decoder12Hz.CodeGroups != 15 || plan.Decoder12Hz.FrameRateHz != 12 {
		t.Fatalf("decoder plan=%+v", plan.Decoder12Hz)
	}
	bytes, err := plan.Talker.KVBytes(128, 4)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 14680064 {
		t.Fatalf("kv bytes=%d", bytes)
	}
}

func TestTransformerPlanRejectsBadKVSizing(t *testing.T) {
	p := TransformerPlan{HiddenSize: 1024, Layers: 1, Heads: 16, KVHeads: 8, HeadDim: 64, VocabSize: 3072, KVFloatsPerToken: 1}
	if err := p.Validate("test"); err == nil {
		t.Fatal("expected KV sizing error")
	}
	if _, err := p.KVBytes(-1, 4); err == nil {
		t.Fatal("expected invalid max sequence error")
	}
}
