package model

import "testing"

func TestNormalizeGemma4TextConfigCompatibilityWrapper(t *testing.T) {
	json := []byte(`{
		"model_type":"gemma4",
		"architectures":["Gemma4ForConditionalGeneration"],
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":5376,
			"num_hidden_layers":60,
			"num_attention_heads":32,
			"num_key_value_heads":16,
			"num_global_key_value_heads":4,
			"head_dim":256,
			"global_head_dim":512,
			"attention_k_eq_v":true,
			"layer_types":["sliding_attention","full_attention"]
		}
	}`)
	cfg, ok := normalizeGemma4TextConfig(json, LlamaConfig{})
	if !ok {
		t.Fatal("normalizeGemma4TextConfig rejected Gemma4 text_config")
	}
	if cfg.ModelType != "gemma4_text" || cfg.HiddenSize != 5376 || cfg.NumGlobalKVHeads != 4 || !cfg.AttentionKEqV {
		t.Fatalf("cfg=%+v", cfg)
	}
	if got := layerKVHeadsForConfig(cfg, 0); got != 16 {
		t.Fatalf("sliding KV heads=%d want 16", got)
	}
	if got := layerKVHeadsForConfig(cfg, 1); got != 4 {
		t.Fatalf("full KV heads=%d want 4", got)
	}
}

func TestGemma4LayerKVDimUsesGlobalKVHeadsForFullAttention(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 16, NumGlobalKVHeads: 4, HeadDim: 256, GlobalHeadDim: 512, LayerTypes: []string{"sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true, HeadDimLocal: 256}, {HasKV: true, HeadDimLocal: 512}},
	}
	if got, err := m.LayerKVDim(0); err != nil || got != 4096 {
		t.Fatalf("sliding LayerKVDim=%d err=%v want 4096", got, err)
	}
	if got, err := m.LayerKVDim(1); err != nil || got != 2048 {
		t.Fatalf("full LayerKVDim=%d err=%v want 2048", got, err)
	}
}
