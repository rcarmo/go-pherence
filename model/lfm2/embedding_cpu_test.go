package lfm2

import (
	"fmt"
	"strings"
	"testing"
)

type fakeLFM2TensorSource map[string]struct {
	data  []float32
	shape []int
}

func (s fakeLFM2TensorSource) GetFloat32(name string) ([]float32, []int, error) {
	t, ok := s[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return append([]float32(nil), t.data...), append([]int(nil), t.shape...), nil
}

func tinyLFM2Config(tied bool) Config {
	return Config{
		ModelType: ModelType, VocabSize: 4, HiddenSize: 2, IntermediateSize: 4,
		NumHiddenLayers: 2, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 2,
		MaxPositionEmbeddings: 16, LayerTypes: []string{"conv", "full_attention"},
		NumDenseLayers: 1, NumExperts: 2, NumExpertsPerTok: 1, MoEIntermediateSize: 2,
		ConvLCache: 2, NormEps: 1e-5, NormTopKProb: true, RoutedScalingFactor: 1,
		TieWordEmbeddings: tied, RoPE: RoPEParameters{Theta: 10000, Type: "default"},
	}
}

func tinyLFM2Plan(t *testing.T, cfg Config, tokens []uint32) RuntimeRequestPlan {
	t.Helper()
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Tokens: tokens, MaxNewTokens: 1, BytesPerFloat: 4})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func TestEmbeddingCPUUsesOwnedTiedWeightsAndFinalNorm(t *testing.T) {
	cfg := tinyLFM2Config(true)
	src := fakeLFM2TensorSource{
		"model.embed_tokens.weight":   {data: []float32{1, 0, 0, 1, 2, 3, -1, 4}, shape: []int{4, 2}},
		"model.embedding_norm.weight": {data: []float32{1, 1}, shape: []int{2}},
	}
	model, err := LoadEmbeddingCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	src["model.embed_tokens.weight"].data[4] = 99
	plan := tinyLFM2Plan(t, cfg, []uint32{2, 1})
	hidden, err := model.Embed(plan, []uint32{2, 1})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 3, 0, 1}
	for i := range want {
		if hidden[i] != want[i] {
			t.Fatalf("hidden=%v want=%v", hidden, want)
		}
	}
	logits, err := model.FinalLogits(hidden[:2])
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != cfg.VocabSize || logits[2] <= logits[1] {
		t.Fatalf("logits=%v", logits)
	}
}

func TestEmbeddingCPUUsesUntiedLMHead(t *testing.T) {
	cfg := tinyLFM2Config(false)
	src := fakeLFM2TensorSource{
		"model.embed_tokens.weight":   {data: make([]float32, 8), shape: []int{4, 2}},
		"model.embedding_norm.weight": {data: []float32{1, 1}, shape: []int{2}},
		"lm_head.weight":              {data: []float32{1, 0, 0, 1, 2, 0, 0, 2}, shape: []int{4, 2}},
	}
	model, err := LoadEmbeddingCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := model.FinalLogits([]float32{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if logits[3] <= logits[1] || logits[2] <= logits[0] {
		t.Fatalf("untied logits=%v", logits)
	}
}

func TestEmbeddingCPURejectsMalformedInputs(t *testing.T) {
	cfg := tinyLFM2Config(true)
	if _, err := LoadEmbeddingCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	if _, err := LoadEmbeddingCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	src := fakeLFM2TensorSource{
		"model.embed_tokens.weight":   {data: make([]float32, 8), shape: []int{2, 4}},
		"model.embedding_norm.weight": {data: []float32{1, 1}, shape: []int{2}},
	}
	if _, err := LoadEmbeddingCPU(src, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
}

func TestRuntimeRequestPlanOwnsTokens(t *testing.T) {
	cfg := tinyLFM2Config(true)
	tokens := []uint32{1, 2}
	plan := tinyLFM2Plan(t, cfg, tokens)
	tokens[0] = 3
	if plan.Tokens[0] != 1 {
		t.Fatalf("plan aliases caller tokens: %v", plan.Tokens)
	}
	plan.Tokens = plan.Tokens[:1]
	if err := plan.Validate(); err == nil {
		t.Fatal("accepted mismatched owned tokens")
	}
}
