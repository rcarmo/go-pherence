package gemma

import (
	"testing"

	"github.com/rcarmo/go-pherence/model/common"
)

func TestNormalizeTextConfigCarries31BFields(t *testing.T) {
	data := []byte(`{
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
	cfg, ok := NormalizeTextConfig(data, common.Config{})
	if !ok {
		t.Fatal("NormalizeTextConfig rejected Gemma4 text_config")
	}
	if cfg.ModelType != "gemma4_text" || cfg.HiddenSize != 5376 || cfg.NumGlobalKVHeads != 4 || !cfg.AttentionKEqV {
		t.Fatalf("cfg=%+v", cfg)
	}
	if got := LayerKVHeads(cfg, 0); got != 16 {
		t.Fatalf("sliding KV heads=%d want 16", got)
	}
	if got := LayerKVHeads(cfg, 1); got != 4 {
		t.Fatalf("full KV heads=%d want 4", got)
	}
}
