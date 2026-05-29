package qwen3tts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type ModelType string

const (
	Base        ModelType = "base"
	CustomVoice ModelType = "custom_voice"
	VoiceDesign ModelType = "voice_design"
)

type SpeakerEncoderConfig struct {
	EncDim     int `json:"enc_dim"`
	SampleRate int `json:"sample_rate"`
}

type ParsedConfig struct {
	ModelType ModelType `json:"model_type"`
	ModelSize string    `json:"model_size"`

	TalkerHiddenSize           int     `json:"talker_hidden_size"`
	TalkerIntermediateSize     int     `json:"talker_intermediate_size"`
	TalkerNumHiddenLayers      int     `json:"talker_num_hidden_layers"`
	TalkerNumAttentionHeads    int     `json:"talker_num_attention_heads"`
	TalkerNumKeyValueHeads     int     `json:"talker_num_key_value_heads"`
	TalkerHeadDim              int     `json:"talker_head_dim"`
	TalkerVocabSize            int     `json:"talker_vocab_size"`
	TalkerTextVocabSize        int     `json:"talker_text_vocab_size"`
	TalkerTextHiddenSize       int     `json:"talker_text_hidden_size"`
	TalkerRMSNormEps           float64 `json:"talker_rms_norm_eps"`
	TalkerRoPETheta            float64 `json:"talker_rope_theta"`
	TalkerMaxPositionEmbedding int     `json:"talker_max_position_embeddings"`
	MRoPESection               [3]int  `json:"mrope_section,omitempty"`
	HasMRoPESection            bool    `json:"has_mrope_section"`

	CPHiddenSize        int     `json:"cp_hidden_size"`
	CPIntermediateSize  int     `json:"cp_intermediate_size"`
	CPNumHiddenLayers   int     `json:"cp_num_hidden_layers"`
	CPNumAttentionHeads int     `json:"cp_num_attention_heads"`
	CPNumKeyValueHeads  int     `json:"cp_num_key_value_heads"`
	CPHeadDim           int     `json:"cp_head_dim"`
	CPVocabSize         int     `json:"cp_vocab_size"`
	CPNumCodeGroups     int     `json:"cp_num_code_groups"`
	CPRMSNormEps        float64 `json:"cp_rms_norm_eps"`
	CPRoPETheta         float64 `json:"cp_rope_theta"`

	SpeakerEncoder *SpeakerEncoderConfig `json:"speaker_encoder,omitempty"`
}

func ParseConfigFile(path string) (ParsedConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ParsedConfig{}, err
	}
	return ParseConfig(data)
}

func ParseConfig(data []byte) (ParsedConfig, error) {
	var v map[string]any
	if err := json.Unmarshal(data, &v); err != nil {
		return ParsedConfig{}, err
	}
	t := obj(v, "talker_config")
	cp := obj(t, "code_predictor_config")
	mt := ModelType(str(v, "tts_model_type", "base"))
	if mt != Base && mt != CustomVoice && mt != VoiceDesign {
		return ParsedConfig{}, fmt.Errorf("unknown Qwen3-TTS model type %q", mt)
	}
	talkerHidden := i(t, "hidden_size", 1024)
	talkerHeads := i(t, "num_attention_heads", 16)
	talkerHeadDim := i(t, "head_dim", derivedHeadDim(talkerHidden, talkerHeads))
	cpHidden := i(cp, "hidden_size", 1024)
	cpHeads := i(cp, "num_attention_heads", 16)
	cpHeadDim := i(cp, "head_dim", derivedHeadDim(cpHidden, cpHeads))
	out := ParsedConfig{
		ModelType:                  mt,
		ModelSize:                  str(v, "tts_model_size", "unknown"),
		TalkerHiddenSize:           talkerHidden,
		TalkerIntermediateSize:     i(t, "intermediate_size", 3072),
		TalkerNumHiddenLayers:      i(t, "num_hidden_layers", 28),
		TalkerNumAttentionHeads:    talkerHeads,
		TalkerNumKeyValueHeads:     i(t, "num_key_value_heads", 8),
		TalkerHeadDim:              talkerHeadDim,
		TalkerVocabSize:            i(t, "vocab_size", CodecVocabSize),
		TalkerTextVocabSize:        i(t, "text_vocab_size", 151936),
		TalkerTextHiddenSize:       i(t, "text_hidden_size", 2048),
		TalkerRMSNormEps:           f(t, "rms_norm_eps", 1e-6),
		TalkerRoPETheta:            f(t, "rope_theta", 1000000.0),
		TalkerMaxPositionEmbedding: i(t, "max_position_embeddings", 32768),
		CPHiddenSize:               cpHidden,
		CPIntermediateSize:         i(cp, "intermediate_size", 3072),
		CPNumHiddenLayers:          i(cp, "num_hidden_layers", 5),
		CPNumAttentionHeads:        cpHeads,
		CPNumKeyValueHeads:         i(cp, "num_key_value_heads", 8),
		CPHeadDim:                  cpHeadDim,
		CPVocabSize:                i(cp, "vocab_size", 2048),
		CPNumCodeGroups:            i(cp, "num_code_groups", 16),
		CPRMSNormEps:               f(cp, "rms_norm_eps", 1e-6),
		CPRoPETheta:                f(cp, "rope_theta", 1000000.0),
	}
	if rs := obj(t, "rope_scaling"); len(rs) > 0 {
		if a, ok := rs["mrope_section"].([]any); ok && len(a) == 3 {
			out.MRoPESection = [3]int{anyInt(a[0], 0), anyInt(a[1], 0), anyInt(a[2], 0)}
			out.HasMRoPESection = true
		}
	}
	if se := obj(v, "speaker_encoder_config"); len(se) > 0 {
		out.SpeakerEncoder = &SpeakerEncoderConfig{EncDim: i(se, "enc_dim", 1024), SampleRate: i(se, "sample_rate", 24000)}
	}
	return out, out.Validate()
}

func (c ParsedConfig) Validate() error {
	if c.TalkerHiddenSize <= 0 || c.TalkerNumAttentionHeads <= 0 || c.TalkerNumKeyValueHeads <= 0 || c.TalkerHeadDim <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS talker attention dims: %+v", c)
	}
	if c.TalkerHiddenSize != c.TalkerNumAttentionHeads*c.TalkerHeadDim {
		return fmt.Errorf("invalid Qwen3-TTS talker head dims: hidden=%d heads=%d head_dim=%d", c.TalkerHiddenSize, c.TalkerNumAttentionHeads, c.TalkerHeadDim)
	}
	if c.TalkerNumKeyValueHeads > c.TalkerNumAttentionHeads || c.TalkerNumAttentionHeads%c.TalkerNumKeyValueHeads != 0 {
		return fmt.Errorf("invalid Qwen3-TTS talker GQA dims: heads=%d kv_heads=%d", c.TalkerNumAttentionHeads, c.TalkerNumKeyValueHeads)
	}
	if c.CPHiddenSize <= 0 || c.CPNumAttentionHeads <= 0 || c.CPNumKeyValueHeads <= 0 || c.CPHeadDim <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS code predictor attention dims: %+v", c)
	}
	if c.CPHiddenSize != c.CPNumAttentionHeads*c.CPHeadDim {
		return fmt.Errorf("invalid Qwen3-TTS code predictor head dims: hidden=%d heads=%d head_dim=%d", c.CPHiddenSize, c.CPNumAttentionHeads, c.CPHeadDim)
	}
	if c.CPNumKeyValueHeads > c.CPNumAttentionHeads || c.CPNumAttentionHeads%c.CPNumKeyValueHeads != 0 {
		return fmt.Errorf("invalid Qwen3-TTS code predictor GQA dims: heads=%d kv_heads=%d", c.CPNumAttentionHeads, c.CPNumKeyValueHeads)
	}
	if c.TalkerNumHiddenLayers <= 0 || c.CPNumHiddenLayers <= 0 || c.CPNumCodeGroups < 2 {
		return fmt.Errorf("invalid Qwen3-TTS layer/codebook counts: talker_layers=%d cp_layers=%d code_groups=%d", c.TalkerNumHiddenLayers, c.CPNumHiddenLayers, c.CPNumCodeGroups)
	}
	return nil
}

func (c ParsedConfig) Label() string {
	size := c.ModelSize
	if size == "0b6" {
		size = "0.6B"
	}
	if size == "1b7" {
		size = "1.7B"
	}
	variant := map[ModelType]string{Base: "Base", CustomVoice: "CustomVoice", VoiceDesign: "VoiceDesign"}[c.ModelType]
	return fmt.Sprintf("%s %s", size, variant)
}

func ReadModelDir(dir string) (ParsedConfig, error) {
	return ParseConfigFile(filepath.Join(dir, "config.json"))
}

func obj(m map[string]any, key string) map[string]any {
	if x, ok := m[key].(map[string]any); ok {
		return x
	}
	return nil
}
func str(m map[string]any, key, def string) string {
	if x, ok := m[key].(string); ok {
		return x
	}
	return def
}
func i(m map[string]any, key string, def int) int {
	if x, ok := m[key]; ok {
		return anyInt(x, def)
	}
	return def
}
func f(m map[string]any, key string, def float64) float64 {
	if x, ok := m[key].(float64); ok {
		return x
	}
	return def
}
func anyInt(x any, def int) int {
	switch v := x.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return def
	}
}

func derivedHeadDim(hidden, heads int) int {
	if hidden > 0 && heads > 0 && hidden%heads == 0 {
		return hidden / heads
	}
	return 0
}
