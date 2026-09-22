package qwen3tts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

type fakeTalkerTensorSource map[string]struct {
	data  []float32
	shape []int
}

func (s fakeTalkerTensorSource) GetFloat32(name string) ([]float32, []int, error) {
	t, ok := s[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return append([]float32(nil), t.data...), append([]int(nil), t.shape...), nil
}

func tinyTalkerConfig() ParsedConfig {
	return ParsedConfig{
		ModelType: CustomVoice, ModelSize: "synthetic",
		TalkerHiddenSize: 4, TalkerIntermediateSize: 4, TalkerNumHiddenLayers: 1,
		TalkerNumAttentionHeads: 1, TalkerNumKeyValueHeads: 1, TalkerHeadDim: 4,
		TalkerVocabSize: CodecVocabSize, TalkerTextVocabSize: 151936, TalkerTextHiddenSize: 4,
		TalkerRMSNormEps: 1e-6, TalkerRoPETheta: 10000, TalkerMaxPositionEmbedding: 128,
		MRoPESection: [3]int{2, 0, 0}, HasMRoPESection: true,
		CPHiddenSize: 4, CPIntermediateSize: 4, CPNumHiddenLayers: 1,
		CPNumAttentionHeads: 1, CPNumKeyValueHeads: 1, CPHeadDim: 4,
		CPVocabSize: 2048, CPNumCodeGroups: 16, CPRMSNormEps: 1e-6, CPRoPETheta: 10000,
	}
}

func tinyTalkerSource(cfg ParsedConfig) fakeTalkerTensorSource {
	s := fakeTalkerTensorSource{}
	add := func(name string, shape []int, data []float32) {
		s[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	identity := func(rows, cols int) []float32 {
		out := make([]float32, rows*cols)
		for i := 0; i < min(rows, cols); i++ {
			out[i*cols+i] = 1
		}
		return out
	}
	text := make([]float32, cfg.TalkerTextVocabSize*cfg.TalkerTextHiddenSize)
	for _, id := range []uint32{IMStart, Assistant, Newline, TTSPad, TTSBOS, 123} {
		base := int(id) * cfg.TalkerTextHiddenSize
		copy(text[base:base+4], []float32{1, 2, 3, 4})
	}
	codec := make([]float32, cfg.TalkerVocabSize*cfg.TalkerHiddenSize)
	copy(codec[int(CodecBOS)*4:int(CodecBOS)*4+4], []float32{1, 0, 0, 0})
	add("talker.model.text_embedding.weight", []int{cfg.TalkerTextVocabSize, cfg.TalkerTextHiddenSize}, text)
	add("talker.model.codec_embedding.weight", []int{cfg.TalkerVocabSize, cfg.TalkerHiddenSize}, codec)
	add("talker.text_projection.linear_fc1.weight", []int{4, 4}, identity(4, 4))
	add("talker.text_projection.linear_fc1.bias", []int{4}, make([]float32, 4))
	add("talker.text_projection.linear_fc2.weight", []int{4, 4}, identity(4, 4))
	add("talker.text_projection.linear_fc2.bias", []int{4}, make([]float32, 4))
	add("talker.model.norm.weight", []int{4}, []float32{1, 1, 1, 1})
	head := make([]float32, cfg.TalkerVocabSize*4)
	copy(head[7*4:8*4], []float32{1, 1, 1, 1})
	copy(head[int(CodecEOS)*4:int(CodecEOS)*4+4], []float32{.1, .1, .1, .1})
	add("talker.codec_head.weight", []int{cfg.TalkerVocabSize, 4}, head)
	prefix := "talker.model.layers.0"
	add(prefix+".input_layernorm.weight", []int{4}, []float32{1, 1, 1, 1})
	add(prefix+".post_attention_layernorm.weight", []int{4}, []float32{1, 1, 1, 1})
	for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		add(prefix+".self_attn."+name+".weight", []int{4, 4}, make([]float32, 16))
	}
	add(prefix+".self_attn.q_norm.weight", []int{4}, []float32{1, 1, 1, 1})
	add(prefix+".self_attn.k_norm.weight", []int{4}, []float32{1, 1, 1, 1})
	for _, name := range []string{"gate_proj", "up_proj", "down_proj"} {
		add(prefix+".mlp."+name+".weight", []int{4, 4}, make([]float32, 16))
	}
	return s
}

func tinyTalkerPlan(t *testing.T, cfg ParsedConfig) RuntimeRequestPlan {
	t.Helper()
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	text = append(text, 456)
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{
		Conditioning: ConditioningRequest{Speaker: Ryan, Language: English},
		Prompt:       PromptIDs{Text: text, Codec: codec}, MaxFrames: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func TestTalkerCPUPrefillGreedyFirstToken(t *testing.T) {
	cfg := tinyTalkerConfig()
	src := tinyTalkerSource(cfg)
	model, err := LoadTalkerCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// The model owns copied weights and must not depend on subsequent source mutation.
	src["talker.codec_head.weight"].data[7*4] = -100
	plan := tinyTalkerPlan(t, cfg)
	result, err := model.Prefill(plan)
	if err != nil {
		t.Fatal(err)
	}
	if result.SemanticToken != 7 || result.PrefillTokens != 10 || len(result.Hidden) != 4 || len(result.Logits) != CodecVocabSize {
		t.Fatalf("result token/prefill/hidden/logits=%d/%d/%d/%d", result.SemanticToken, result.PrefillTokens, len(result.Hidden), len(result.Logits))
	}
	got, err := model.ForwardSemantic(plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 7 {
		t.Fatalf("semantic=%v", got)
	}
	for i, v := range result.Hidden {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("hidden[%d]=%v", i, v)
		}
	}
}

func TestTalkerCPUOfficialQueryWidthDiffersFromHidden(t *testing.T) {
	cfg := tinyTalkerConfig()
	cfg.TalkerNumAttentionHeads = 2
	cfg.TalkerNumKeyValueHeads = 1
	cfg.MRoPESection = [3]int{2, 0, 0}
	src := tinyTalkerSource(cfg)
	prefix := "talker.model.layers.0.self_attn."
	query := make([]float32, 8*4)
	output := make([]float32, 4*8)
	src[prefix+"q_proj.weight"] = struct {
		data  []float32
		shape []int
	}{query, []int{8, 4}}
	src[prefix+"o_proj.weight"] = struct {
		data  []float32
		shape []int
	}{output, []int{4, 8}}
	model, err := LoadTalkerCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Prefill(tinyTalkerPlan(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	if result.SemanticToken != 7 {
		t.Fatalf("token=%d", result.SemanticToken)
	}
}

func TestTalkerCPUPrefillRejectsForgedControlToken(t *testing.T) {
	cfg := tinyTalkerConfig()
	model, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	plan := tinyTalkerPlan(t, cfg)
	plan.Prompt.Codec[4] = 1
	if _, err := model.Prefill(plan); err == nil || !strings.Contains(err.Error(), "codec control") {
		t.Fatalf("forged control error=%v", err)
	}
}

func TestLoadTalkerCPUFromDirRejectsMissingCheckpoint(t *testing.T) {
	cfg := tinyTalkerConfig()
	if _, err := LoadTalkerCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
}

func TestTalkerCPULoadRejectsShapeAndMissingTensor(t *testing.T) {
	cfg := tinyTalkerConfig()
	if _, err := LoadTalkerCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil tensor source")
	}
	src := tinyTalkerSource(cfg)
	bad := src["talker.codec_head.weight"]
	bad.shape = []int{4, CodecVocabSize}
	src["talker.codec_head.weight"] = bad
	if _, err := LoadTalkerCPU(src, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
	delete(src, "talker.codec_head.weight")
	if _, err := LoadTalkerCPU(src, cfg); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("missing error=%v", err)
	}
}

func TestTalkerCPUSuppressionKeepsEOSAndRejectsNonfinite(t *testing.T) {
	logits := make([]float32, CodecVocabSize)
	logits[2051] = float32(math.NaN()) // suppressed values are ignored
	logits[CodecEOS] = 2
	logits[42] = 1
	got, err := greedyTalkerToken(logits, CodecEOS)
	if err != nil || got != CodecEOS {
		t.Fatalf("got=%d err=%v", got, err)
	}
	logits[1] = float32(math.NaN())
	if _, err := greedyTalkerToken(logits, CodecEOS); err == nil {
		t.Fatal("accepted NaN logit")
	}
}

func TestRuntimeRequestPlanOwnsPromptIDs(t *testing.T) {
	cfg := tinyTalkerConfig()
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	text[0], codec[0] = 0, 0
	if plan.Prompt.Text[0] != IMStart || plan.Prompt.Codec[0] != CodecThink {
		t.Fatalf("plan prompt aliases caller: %+v", plan.Prompt)
	}
	plan.Prompt.Text = plan.Prompt.Text[:len(plan.Prompt.Text)-1]
	if err := plan.Validate(); err == nil {
		t.Fatal("accepted mismatched owned prompt")
	}
}
