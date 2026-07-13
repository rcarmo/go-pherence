// Package mosstranscribe implements the native MOSS-Transcribe-Diarize model.
package mosstranscribe

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/models/whisper"
)

const (
	UpstreamModelType      = "moss_transcribe_diarize"
	UpstreamArchitecture   = "MossTranscribeDiarizeForConditionalGeneration"
	GenerationEOSTokenID   = 151645
	GenerationMaxNewTokens = 5120
)

// Config is the checkpoint architecture contract used by the native runtime.
type Config struct {
	Architectures  []string    `json:"architectures"`
	ModelType      string      `json:"model_type"`
	DType          string      `json:"dtype"`
	Text           TextConfig  `json:"text_config"`
	Audio          AudioConfig `json:"audio_config"`
	AudioTokenID   int         `json:"audio_token_id"`
	AudioMergeSize int         `json:"audio_merge_size"`
	AdaptorInput   int         `json:"adaptor_input_dim"`
	TieEmbeddings  bool        `json:"tie_word_embeddings"`
	PadTokenID     int         `json:"pad_token_id"`
}

type TextConfig struct {
	ModelType         string   `json:"model_type"`
	VocabSize         int      `json:"vocab_size"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumLayers         int      `json:"num_hidden_layers"`
	NumHeads          int      `json:"num_attention_heads"`
	NumKVHeads        int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	HiddenAct         string   `json:"hidden_act"`
	MaxPositions      int      `json:"max_position_embeddings"`
	RMSNormEps        float64  `json:"rms_norm_eps"`
	RopeTheta         float64  `json:"rope_theta"`
	AttentionBias     bool     `json:"attention_bias"`
	AttentionDropout  float64  `json:"attention_dropout"`
	LayerTypes        []string `json:"layer_types"`
	TieWordEmbeddings bool     `json:"tie_word_embeddings"`
	PadTokenID        int      `json:"pad_token_id"`
}

type AudioConfig struct {
	ModelType        string  `json:"model_type"`
	NumMelBins       int     `json:"num_mel_bins"`
	DModel           int     `json:"d_model"`
	NumLayers        int     `json:"encoder_layers"`
	NumHeads         int     `json:"encoder_attention_heads"`
	FFNDim           int     `json:"encoder_ffn_dim"`
	MaxSourcePos     int     `json:"max_source_positions"`
	Activation       string  `json:"activation_function"`
	Dropout          float64 `json:"dropout"`
	AttentionDropout float64 `json:"attention_dropout"`
}

// WhisperConfig maps the audio-only checkpoint contract onto the reusable
// native Whisper encoder. MOSS does not use Whisper's text decoder.
func (c Config) WhisperConfig() whisper.Config {
	return whisper.Config{
		NumMelBins:       c.Audio.NumMelBins,
		MaxLength:        c.Audio.MaxSourcePos * 2,
		EncoderLayers:    c.Audio.NumLayers,
		EncoderDModel:    c.Audio.DModel,
		EncoderHeads:     c.Audio.NumHeads,
		EncoderFFNDim:    c.Audio.FFNDim,
		HeadDim:          c.Audio.DModel / c.Audio.NumHeads,
		MaxDecoderLength: 0,
	}
}

// LoadConfig reads and validates a Hugging Face config.json.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("decode MOSS config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// Validate rejects architectures that the native implementation cannot execute exactly.
func (c Config) Validate() error {
	if c.ModelType != UpstreamModelType {
		return fmt.Errorf("unsupported MOSS model_type %q", c.ModelType)
	}
	if len(c.Architectures) != 1 || c.Architectures[0] != UpstreamArchitecture {
		return fmt.Errorf("unsupported MOSS architectures %v", c.Architectures)
	}
	if c.DType != "bfloat16" {
		return fmt.Errorf("unsupported MOSS dtype %q", c.DType)
	}
	if c.AudioTokenID < 0 || c.PadTokenID < 0 || c.AudioMergeSize != 4 {
		return fmt.Errorf("invalid MOSS token/merge contract audio_token=%d pad_token=%d merge=%d", c.AudioTokenID, c.PadTokenID, c.AudioMergeSize)
	}
	if c.Audio.ModelType != "whisper" || c.Audio.NumMelBins != 80 || c.Audio.DModel != 1024 || c.Audio.NumLayers != 24 || c.Audio.NumHeads != 16 || c.Audio.FFNDim != 4096 || c.Audio.MaxSourcePos != 1500 || c.Audio.Activation != "gelu" {
		return fmt.Errorf("unsupported MOSS Whisper encoder config: %+v", c.Audio)
	}
	if c.AdaptorInput != c.Audio.DModel*c.AudioMergeSize {
		return fmt.Errorf("MOSS adaptor input=%d, want audio width %d * merge %d", c.AdaptorInput, c.Audio.DModel, c.AudioMergeSize)
	}
	if c.Text.ModelType != "qwen3" || c.Text.VocabSize != 151936 || c.Text.HiddenSize != 1024 || c.Text.IntermediateSize != 3072 || c.Text.NumLayers != 28 || c.Text.NumHeads != 16 || c.Text.NumKVHeads != 8 || c.Text.HeadDim != 128 || c.Text.HiddenAct != "silu" || c.Text.MaxPositions != 131072 || c.Text.RopeTheta != 1_000_000 || c.Text.AttentionBias || !c.Text.TieWordEmbeddings || !c.TieEmbeddings {
		return fmt.Errorf("unsupported MOSS Qwen3 decoder config: %+v", c.Text)
	}
	if c.Text.RMSNormEps != 1e-6 || c.Audio.Dropout != 0 || c.Audio.AttentionDropout != 0 || c.Text.AttentionDropout != 0 {
		return fmt.Errorf("unsupported MOSS norm/dropout contract")
	}
	if len(c.Text.LayerTypes) != c.Text.NumLayers {
		return fmt.Errorf("MOSS layer_types=%d, want %d", len(c.Text.LayerTypes), c.Text.NumLayers)
	}
	for i, typ := range c.Text.LayerTypes {
		if typ != "full_attention" {
			return fmt.Errorf("MOSS layer %d type %q is unsupported", i, typ)
		}
	}
	return nil
}
