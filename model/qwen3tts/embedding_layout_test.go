package qwen3tts

import "testing"

func TestEmbeddingLayout(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"talker_config":{
			"hidden_size":1024,
			"num_attention_heads":16,
			"head_dim":64,
			"vocab_size":3072,
			"text_vocab_size":151936,
			"text_hidden_size":2048,
			"code_predictor_config":{
				"hidden_size":1024,
				"num_attention_heads":16,
				"head_dim":64,
				"vocab_size":2048
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewEmbeddingLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if layout.TextEmbeddingFloats != 311164928 || layout.TextProjectionFloats != 2097152 || layout.CodecHeadFloats != 3145728 || layout.CodecEmbeddingFloats != 2097152 {
		t.Fatalf("layout=%+v", layout)
	}
	if layout.TotalBridgeFloats != 318504960 {
		t.Fatalf("total=%d", layout.TotalBridgeFloats)
	}
}

func TestEmbeddingLayoutRejectsMalformed(t *testing.T) {
	bad := EmbeddingLayout{TalkerTextVocabSize: 151936, TalkerTextHiddenSize: 2048, TalkerHiddenSize: 1024, TalkerCodecVocabSize: 3072, CodePredictorHiddenSize: 1024, CodePredictorVocabSize: 2048, TextEmbeddingFloats: 1, TextProjectionFloats: 2097152, CodecHeadFloats: 3145728, CodecEmbeddingFloats: 2097152, TotalBridgeFloats: 318504960}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected text embedding error")
	}
	bad = EmbeddingLayout{TalkerTextVocabSize: 151936, TalkerTextHiddenSize: 2048, TalkerHiddenSize: 1024, TalkerCodecVocabSize: 3072, CodePredictorHiddenSize: 1024, CodePredictorVocabSize: 2048, TextEmbeddingFloats: 311164928, TextProjectionFloats: 1, CodecHeadFloats: 3145728, CodecEmbeddingFloats: 2097152, TotalBridgeFloats: 318504960}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected text projection error")
	}
	bad = EmbeddingLayout{TalkerTextVocabSize: 151936, TalkerTextHiddenSize: 2048, TalkerHiddenSize: 1024, TalkerCodecVocabSize: 3072, CodePredictorHiddenSize: 1024, CodePredictorVocabSize: 2048, TextEmbeddingFloats: 311164928, TextProjectionFloats: 2097152, CodecHeadFloats: 3145728, CodecEmbeddingFloats: 2097152, TotalBridgeFloats: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected total error")
	}
}
