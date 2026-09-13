package omnivoice

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// Config is the typed OmniVoice model config.json shape needed to derive the
// checkpoint tensor layout.
type Config struct {
	Architectures        []string  `json:"architectures"`
	AudioCodebookWeights []int     `json:"audio_codebook_weights"`
	AudioMaskID          int       `json:"audio_mask_id"`
	AudioVocabSize       int       `json:"audio_vocab_size"`
	BOSTokenID           *int      `json:"bos_token_id"`
	DType                string    `json:"dtype"`
	EOSTokenID           int       `json:"eos_token_id"`
	LLMConfig            LLMConfig `json:"llm_config"`
	ModelType            string    `json:"model_type"`
	NumAudioCodebook     int       `json:"num_audio_codebook"`
	PadTokenID           *int      `json:"pad_token_id"`
	TransformersVersion  string    `json:"transformers_version"`
}

// LLMConfig captures the nested Qwen3 decoder config used by OmniVoice.
type LLMConfig struct {
	NameOrPath            string         `json:"_name_or_path"`
	Architectures         []string       `json:"architectures"`
	AttentionBias         bool           `json:"attention_bias"`
	AttentionDropout      float64        `json:"attention_dropout"`
	BOSTokenID            *int           `json:"bos_token_id"`
	ChunkSizeFeedForward  int            `json:"chunk_size_feed_forward"`
	DType                 string         `json:"dtype"`
	EOSTokenID            int            `json:"eos_token_id"`
	HeadDim               int            `json:"head_dim"`
	HiddenAct             string         `json:"hidden_act"`
	HiddenSize            int            `json:"hidden_size"`
	InitializerRange      float64        `json:"initializer_range"`
	IntermediateSize      int            `json:"intermediate_size"`
	LayerTypes            []string       `json:"layer_types"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	MaxWindowLayers       int            `json:"max_window_layers"`
	ModelType             string         `json:"model_type"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	RopeParameters        RopeParameters `json:"rope_parameters"`
	SlidingWindow         *int           `json:"sliding_window"`
	TieWordEmbeddings     bool           `json:"tie_word_embeddings"`
	UseCache              bool           `json:"use_cache"`
	UseSlidingWindow      bool           `json:"use_sliding_window"`
	VocabSize             int            `json:"vocab_size"`
}

// RopeParameters is the subset of rope_parameters needed for validation.
type RopeParameters struct {
	RopeTheta float64 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

// LoadConfig loads and validates an OmniVoice config. path may be either a
// config.json path or a model directory containing config.json.
func LoadConfig(path string) (Config, error) {
	configPath, err := resolveConfigPath(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if _, err := loaderconfig.ReadJSON(configPath, &cfg); err != nil {
		return Config{}, fmt.Errorf("omnivoice: read config %s: %w", configPath, err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("omnivoice: validate config %s: %w", configPath, err)
	}
	return cfg, nil
}

// Validate checks whether the config matches the OmniVoice/Qwen3 layout this
// loader understands.
func (c Config) Validate() error {
	if !containsString(c.Architectures, "OmniVoice") {
		return fmt.Errorf("architectures must include OmniVoice")
	}
	if c.ModelType != "omnivoice" {
		return fmt.Errorf("model_type=%q, want %q", c.ModelType, "omnivoice")
	}
	if c.NumAudioCodebook <= 0 {
		return fmt.Errorf("num_audio_codebook=%d must be > 0", c.NumAudioCodebook)
	}
	if c.AudioVocabSize <= 0 {
		return fmt.Errorf("audio_vocab_size=%d must be > 0", c.AudioVocabSize)
	}
	if c.AudioMaskID < 0 || c.AudioMaskID >= c.AudioVocabSize {
		return fmt.Errorf("audio_mask_id=%d outside [0,%d)", c.AudioMaskID, c.AudioVocabSize)
	}
	if len(c.AudioCodebookWeights) != c.NumAudioCodebook {
		return fmt.Errorf("audio_codebook_weights has %d entries, want %d", len(c.AudioCodebookWeights), c.NumAudioCodebook)
	}
	for i, w := range c.AudioCodebookWeights {
		if w <= 0 {
			return fmt.Errorf("audio_codebook_weights[%d]=%d must be > 0", i, w)
		}
	}
	if err := c.LLMConfig.Validate(); err != nil {
		return fmt.Errorf("llm_config: %w", err)
	}
	return nil
}

// Validate checks whether the nested decoder config matches the Qwen3 layout
// emitted inside current OmniVoice checkpoints.
func (c LLMConfig) Validate() error {
	if !containsString(c.Architectures, "Qwen3ForCausalLM") {
		return fmt.Errorf("architectures must include Qwen3ForCausalLM")
	}
	if c.ModelType != "qwen3" {
		return fmt.Errorf("model_type=%q, want %q", c.ModelType, "qwen3")
	}
	if c.HiddenSize <= 0 {
		return fmt.Errorf("hidden_size=%d must be > 0", c.HiddenSize)
	}
	if c.IntermediateSize <= 0 {
		return fmt.Errorf("intermediate_size=%d must be > 0", c.IntermediateSize)
	}
	if c.HeadDim <= 0 {
		return fmt.Errorf("head_dim=%d must be > 0", c.HeadDim)
	}
	if c.NumAttentionHeads <= 0 {
		return fmt.Errorf("num_attention_heads=%d must be > 0", c.NumAttentionHeads)
	}
	if c.NumHiddenLayers <= 0 {
		return fmt.Errorf("num_hidden_layers=%d must be > 0", c.NumHiddenLayers)
	}
	if c.NumKeyValueHeads <= 0 {
		return fmt.Errorf("num_key_value_heads=%d must be > 0", c.NumKeyValueHeads)
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("attention heads must be divisible by KV heads")
	}
	if c.NumKeyValueHeads > c.NumAttentionHeads {
		return fmt.Errorf("num_key_value_heads=%d exceeds num_attention_heads=%d", c.NumKeyValueHeads, c.NumAttentionHeads)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab_size=%d must be > 0", c.VocabSize)
	}
	if len(c.LayerTypes) > 0 {
		if len(c.LayerTypes) != c.NumHiddenLayers {
			return fmt.Errorf("layer_types has %d entries, want %d", len(c.LayerTypes), c.NumHiddenLayers)
		}
		for i, layerType := range c.LayerTypes {
			if layerType != "full_attention" {
				return fmt.Errorf("layer_types[%d]=%q unsupported (want full_attention)", i, layerType)
			}
		}
	}
	if c.RopeParameters.RopeTheta <= 0 || math.IsNaN(c.RopeParameters.RopeTheta) || math.IsInf(c.RopeParameters.RopeTheta, 0) {
		return fmt.Errorf("rope_parameters.rope_theta=%g must be > 0", c.RopeParameters.RopeTheta)
	}
	if c.RMSNormEps <= 0 || math.IsNaN(c.RMSNormEps) || math.IsInf(c.RMSNormEps, 0) {
		return fmt.Errorf("rms_norm_eps must be finite and positive")
	}
	if c.RopeParameters.RopeType == "" {
		return fmt.Errorf("rope_parameters.rope_type must be set")
	}
	return nil
}

func resolveConfigPath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("omnivoice: empty config path")
	}
	fi, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("omnivoice: stat %s: %w", path, err)
	}
	if fi.IsDir() {
		return filepath.Join(path, "config.json"), nil
	}
	return path, nil
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
