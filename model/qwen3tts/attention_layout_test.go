package qwen3tts

import "testing"

func TestAttentionLayouts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"talker_config":{
			"hidden_size":1024,
			"num_hidden_layers":28,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"head_dim":64,
			"rms_norm_eps":0.000001,
			"rope_theta":1000000,
			"max_position_embeddings":32768,
			"code_predictor_config":{
				"hidden_size":1024,
				"num_hidden_layers":5,
				"num_attention_heads":16,
				"num_key_value_heads":8,
				"head_dim":64,
				"rms_norm_eps":0.000001,
				"rope_theta":1000000
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	talker, err := NewTalkerAttentionLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if talker.Name != "talker" || talker.QueriesPerKV != 2 || talker.MaxPositionEmbeddings != 32768 || talker.RoPETheta != 1000000 {
		t.Fatalf("talker=%+v", talker)
	}
	if err := talker.ValidatePosition(32767); err != nil {
		t.Fatal(err)
	}
	if err := talker.ValidatePosition(32768); err == nil {
		t.Fatal("expected max position error")
	}
	kvFloats, err := talker.KVFloatsPerToken()
	if err != nil {
		t.Fatal(err)
	}
	if kvFloats != 28672 {
		t.Fatalf("talker kv floats=%d", kvFloats)
	}
	kvBytes, err := talker.KVBytes(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if kvBytes != 114688 {
		t.Fatalf("talker kv bytes=%d", kvBytes)
	}
	cp, err := NewCodePredictorAttentionLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cp.Name != "code_predictor" || cp.QueriesPerKV != 2 || cp.MaxPositionEmbeddings != 0 {
		t.Fatalf("cp=%+v", cp)
	}
	if err := cp.ValidatePosition(1_000_000); err != nil {
		t.Fatal(err)
	}
	cpKVFloats, err := cp.KVFloatsPerToken()
	if err != nil {
		t.Fatal(err)
	}
	if cpKVFloats != 5120 {
		t.Fatalf("cp kv floats=%d", cpKVFloats)
	}
}

func TestAttentionLayoutRejectsMalformed(t *testing.T) {
	bad := AttentionLayout{Name: "talker", HiddenSize: 1024, Layers: 1, Heads: 16, KVHeads: 8, HeadDim: 64, QueriesPerKV: 1, RoPETheta: 1, RMSNormEps: 1e-6}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected queries/kv error")
	}
	bad = AttentionLayout{Name: "talker", HiddenSize: 1024, Layers: 1, Heads: 16, KVHeads: 8, HeadDim: 64, QueriesPerKV: 2, RoPETheta: 0, RMSNormEps: 1e-6}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected rope theta error")
	}
	bad = AttentionLayout{Name: "talker", HiddenSize: 1024, Layers: 1, Heads: 16, KVHeads: 8, HeadDim: 64, QueriesPerKV: 2, RoPETheta: 1, RMSNormEps: 1e-6, MaxPositionEmbeddings: -1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected max position error")
	}
	bad = AttentionLayout{Name: "talker", HiddenSize: 1024, Layers: 1, Heads: 16, KVHeads: 8, HeadDim: 64, QueriesPerKV: 2, RoPETheta: 1, RMSNormEps: 1e-6}
	if err := bad.ValidatePosition(-1); err == nil {
		t.Fatal("expected negative position error")
	}
	if _, err := bad.KVBytes(-1, 2); err == nil {
		t.Fatal("expected KV sizing error")
	}
}
