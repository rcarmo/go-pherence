package qwen3tts

import "testing"

func TestFFNLayouts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"talker_config":{
			"hidden_size":1024,
			"intermediate_size":3072,
			"num_hidden_layers":28,
			"num_attention_heads":16,
			"head_dim":64,
			"code_predictor_config":{
				"hidden_size":1024,
				"intermediate_size":3072,
				"num_hidden_layers":5,
				"num_attention_heads":16,
				"head_dim":64
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	talker, err := NewTalkerFFNLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if talker.Name != "talker" || talker.GateProjectionFloats != 3145728 || talker.FloatsPerLayer != 9437184 || talker.TotalFloats != 264241152 {
		t.Fatalf("talker=%+v", talker)
	}
	cp, err := NewCodePredictorFFNLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cp.Name != "code_predictor" || cp.FloatsPerLayer != 9437184 || cp.TotalFloats != 47185920 {
		t.Fatalf("cp=%+v", cp)
	}
}

func TestFFNLayoutRejectsMalformed(t *testing.T) {
	bad := FFNLayout{Name: "talker", HiddenSize: 1024, IntermediateSize: 3072, Layers: 28, GateProjectionFloats: 1, UpProjectionFloats: 3145728, DownProjectionFloats: 3145728, FloatsPerLayer: 9437184, TotalFloats: 264241152}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected projection float error")
	}
	bad = FFNLayout{Name: "talker", HiddenSize: 1024, IntermediateSize: 3072, Layers: 28, GateProjectionFloats: 3145728, UpProjectionFloats: 3145728, DownProjectionFloats: 3145728, FloatsPerLayer: 1, TotalFloats: 264241152}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected floats/layer error")
	}
	bad = FFNLayout{Name: "talker", HiddenSize: 1024, IntermediateSize: 3072, Layers: 28, GateProjectionFloats: 3145728, UpProjectionFloats: 3145728, DownProjectionFloats: 3145728, FloatsPerLayer: 9437184, TotalFloats: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected total float error")
	}
}
