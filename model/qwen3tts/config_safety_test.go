package qwen3tts

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestConfigRejectsInvalidNumericMetadata(t *testing.T) {
	for _, raw := range []string{
		`null`, `{} {}`, `{"talker_config":[]}`, `{"talker_config":{"code_predictor_config":"bad"}}`,
		`{"talker_config":{"num_hidden_layers":1.5}}`, `{"talker_config":{"num_hidden_layers":"28"}}`,
		`{"talker_config":{"num_hidden_layers":9223372036854775808}}`, `{"talker_config":{"rms_norm_eps":"1e-6"}}`,
		`{"talker_config":{"intermediate_size":0}}`, `{"talker_config":{"text_vocab_size":-1}}`,
		`{"talker_config":{"max_position_embeddings":-1}}`, `{"speaker_encoder_config":{"sample_rate":0}}`,
		`{"talker_config":{"rope_scaling":{"mrope_section":[1.5,2,3]}}}`,
		`{"talker_config":{"rope_scaling":{"mrope_section":[1,2]}}}`,
		`{"talker_config":{"code_predictor_config":{"num_code_groups":9223372036854775807}}}`,
	} {
		if _, err := ParseConfig([]byte(raw)); err == nil {
			t.Errorf("accepted %s", raw)
		}
	}
}

func TestConfigPreservesExactIntegerMetadata(t *testing.T) {
	// This value rounds if decoded through float64, but is representable on 64-bit.
	if strconv.IntSize < 64 {
		t.Skip("requires 64-bit int")
	}
	raw := []byte(`{"talker_config":{"text_vocab_size":9007199254740993}}`)
	c, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(c.TalkerTextVocabSize) != "9007199254740993" {
		t.Fatal("integer rounded", c.TalkerTextVocabSize)
	}
}

func TestTensorShapesUseNestedCodePredictorDimensions(t *testing.T) {
	c := sizingConfig(t)
	c.CPHiddenSize = 512
	c.CPHeadDim = 32
	c.CPIntermediateSize = 1536
	infos := map[string]safetensors.TensorInfo{
		"talker.code_predictor.model.layers.0.self_attn.q_proj.weight": {Shape: []int{512, 512}},
		"talker.code_predictor.model.layers.0.self_attn.k_proj.weight": {Shape: []int{256, 512}},
		"talker.code_predictor.model.layers.0.mlp.up_proj.weight":      {Shape: []int{1536, 512}},
	}
	if v := ValidateTensorShapes(c, infos); !v.Valid {
		t.Fatal(v)
	}
	infos["talker.code_predictor.model.layers.0.self_attn.q_proj.weight"] = safetensors.TensorInfo{Shape: []int{1024, 1024}}
	if v := ValidateTensorShapes(c, infos); v.Valid {
		t.Fatal("talker-sized CP matrix accepted")
	}
}

func TestTensorShapesRejectMalformedConfigAndEqualWidthMatrices(t *testing.T) {
	c := sizingConfig(t)
	c.TalkerTextHiddenSize = c.TalkerHiddenSize
	c.TalkerVocabSize = c.TalkerHiddenSize
	for _, name := range []string{"talker.text_projection.weight", "talker.codec_head.weight"} {
		if v := ValidateTensorShapes(c, map[string]safetensors.TensorInfo{name: {Shape: []int{c.TalkerHiddenSize, 1}}}); v.Valid {
			t.Fatal("wrong square matrix accepted", name)
		}
	}
	c.TalkerHeadDim = int(^uint(0) >> 1)
	if v := ValidateTensorShapes(c, nil); v.Valid {
		t.Fatal("invalid config accepted")
	}
}

func TestFrameConstructorBoundsGroupAllocation(t *testing.T) {
	c := sizingConfig(t)
	c.CPNumCodeGroups = int(^uint(0) >> 1)
	if _, err := NewAcousticFrameLayout(c); err == nil {
		t.Fatal("oversized group allocation accepted")
	}
	c.CPNumCodeGroups = maxCodeGroups
	if l, err := NewAcousticFrameLayout(c); err != nil || len(l.AcousticGroups) != maxCodeGroups-1 {
		t.Fatal(l, err)
	}
}
