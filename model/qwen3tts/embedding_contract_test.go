package qwen3tts

import "testing"

func TestRuntimePlanAppliesEmbeddingContracts(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":3072,"text_vocab_size":151936,"text_hidden_size":2048,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Talker.HiddenSize != plan.EmbeddingLayout.TalkerHiddenSize || plan.Talker.VocabSize != plan.EmbeddingLayout.TalkerCodecVocabSize {
		t.Fatalf("talker embedding contract not applied: talker=%+v embedding=%+v", plan.Talker, plan.EmbeddingLayout)
	}
	if plan.CodePredictor.HiddenSize != plan.EmbeddingLayout.CodePredictorHiddenSize || plan.CodePredictor.VocabSize != plan.EmbeddingLayout.CodePredictorVocabSize {
		t.Fatalf("code predictor embedding contract not applied: cp=%+v embedding=%+v", plan.CodePredictor, plan.EmbeddingLayout)
	}
	plan.Talker.VocabSize++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected talker embedding layout mismatch")
	}
	plan.Talker.VocabSize = plan.EmbeddingLayout.TalkerCodecVocabSize
	plan.CodePredictor.HiddenSize++
	if err := plan.Validate(); err == nil {
		t.Fatal("expected code predictor embedding layout mismatch")
	}
}
