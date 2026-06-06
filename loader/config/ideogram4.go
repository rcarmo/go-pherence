package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Ideogram4ModelIndex describes the Diffusers model_index.json published for
// ideogram-ai/ideogram-4-fp8. Runtime generation needs a Qwen3-VL text encoder,
// conditional and unconditional Ideogram4 transformers, scheduler, and VAE; this
// config layer provides local-asset validation and readiness reporting.
type Ideogram4ModelIndex struct {
	ClassName                string   `json:"_class_name"`
	DiffusersVersion         string   `json:"_diffusers_version"`
	Scheduler                []string `json:"scheduler"`
	TextEncoder              []string `json:"text_encoder"`
	Tokenizer                []string `json:"tokenizer"`
	Transformer              []string `json:"transformer"`
	UnconditionalTransformer []string `json:"unconditional_transformer"`
	VAE                      []string `json:"vae"`
}

type Ideogram4TransformerConfig struct {
	ClassName         string  `json:"_class_name"`
	EmbDim            int     `json:"emb_dim"`
	NumLayers         int     `json:"num_layers"`
	NumHeads          int     `json:"num_heads"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	AttentionHeadDim  int     `json:"attention_head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	AdaLNDim          int     `json:"adaln_dim"`
	InChannels        int     `json:"in_channels"`
	LLMFeaturesDim    int     `json:"llm_features_dim"`
	RopeTheta         int     `json:"rope_theta"`
	MRoPESection      []int   `json:"mrope_section"`
	NormEps           float64 `json:"norm_eps"`
	Quantization      any     `json:"quantization_config,omitempty"`
}

type Ideogram4VAEConfig struct {
	ClassName       string  `json:"_class_name"`
	InChannels      int     `json:"in_channels"`
	OutChannels     int     `json:"out_channels"`
	LatentChannels  int     `json:"latent_channels"`
	ScalingFactor   float64 `json:"scaling_factor"`
	ShiftFactor     float64 `json:"shift_factor"`
	SampleSize      int     `json:"sample_size"`
	BlockOutChannel []int   `json:"block_out_channels"`
}

type Ideogram4TextEncoderConfig struct {
	ModelType         string                  `json:"model_type"`
	Architectures     []string                `json:"architectures"`
	HiddenSize        int                     `json:"hidden_size"`
	NumHiddenLayers   int                     `json:"num_hidden_layers"`
	NumAttentionHeads int                     `json:"num_attention_heads"`
	NumKeyValueHeads  int                     `json:"num_key_value_heads"`
	VocabSize         int                     `json:"vocab_size"`
	TextConfig        *Ideogram4TextSubConfig `json:"text_config"`
}

type Ideogram4TextSubConfig struct {
	ModelType         string `json:"model_type"`
	HiddenSize        int    `json:"hidden_size"`
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	NumAttentionHeads int    `json:"num_attention_heads"`
	NumKeyValueHeads  int    `json:"num_key_value_heads"`
	VocabSize         int    `json:"vocab_size"`
}

type Ideogram4SchedulerConfig struct {
	ClassName         string  `json:"_class_name"`
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	Shift             float64 `json:"shift"`
}

type Ideogram4Config struct {
	Root                     string                     `json:"root"`
	ModelIndex               Ideogram4ModelIndex        `json:"model_index"`
	Transformer              Ideogram4TransformerConfig `json:"transformer"`
	UnconditionalTransformer Ideogram4TransformerConfig `json:"unconditional_transformer"`
	VAE                      Ideogram4VAEConfig         `json:"vae"`
	TextEncoder              Ideogram4TextEncoderConfig `json:"text_encoder"`
	Scheduler                Ideogram4SchedulerConfig   `json:"scheduler"`
}

type Ideogram4Summary struct {
	Pipeline                 string `json:"pipeline"`
	Transformer              string `json:"transformer"`
	UnconditionalTransformer string `json:"unconditional_transformer"`
	TextEncoder              string `json:"text_encoder"`
	Tokenizer                string `json:"tokenizer"`
	Scheduler                string `json:"scheduler"`
	VAE                      string `json:"vae"`
	EmbDim                   int    `json:"emb_dim"`
	Layers                   int    `json:"layers"`
	Heads                    int    `json:"heads"`
	HeadDim                  int    `json:"head_dim"`
	IntermediateSize         int    `json:"intermediate_size"`
	AdaLNDim                 int    `json:"adaln_dim"`
	InChannels               int    `json:"in_channels"`
	LLMFeaturesDim           int    `json:"llm_features_dim"`
	MRoPESection             []int  `json:"mrope_section"`
	ActivationLayers         []int  `json:"activation_layers"`
	TextHidden               int    `json:"text_hidden"`
	TextLayers               int    `json:"text_layers"`
	VocabSize                int    `json:"vocab_size"`
	RuntimeReady             bool   `json:"runtime_ready"`
	RuntimeNote              string `json:"runtime_note"`
}

var Ideogram4Qwen3VLActivationLayers = []int{0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35}

func ReadIdeogram4Config(root string) (Ideogram4Config, error) {
	var cfg Ideogram4Config
	cfg.Root = root
	if err := readIdeogram4JSON(filepath.Join(root, "model_index.json"), &cfg.ModelIndex); err != nil {
		return cfg, err
	}
	if err := readIdeogram4JSON(filepath.Join(root, "transformer", "config.json"), &cfg.Transformer); err != nil {
		return cfg, err
	}
	if err := readIdeogram4JSON(filepath.Join(root, "unconditional_transformer", "config.json"), &cfg.UnconditionalTransformer); err != nil {
		return cfg, err
	}
	if err := readIdeogram4JSON(filepath.Join(root, "vae", "config.json"), &cfg.VAE); err != nil {
		return cfg, err
	}
	if err := readIdeogram4JSON(filepath.Join(root, "text_encoder", "config.json"), &cfg.TextEncoder); err != nil {
		return cfg, err
	}
	if err := readIdeogram4JSON(filepath.Join(root, "scheduler", "scheduler_config.json"), &cfg.Scheduler); err != nil {
		return cfg, err
	}
	normalizeIdeogram4Transformer(&cfg.Transformer)
	normalizeIdeogram4Transformer(&cfg.UnconditionalTransformer)
	return cfg, ValidateIdeogram4Config(cfg)
}

func readIdeogram4JSON(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func ValidateIdeogram4Config(cfg Ideogram4Config) error {
	normalizeIdeogram4Transformer(&cfg.Transformer)
	normalizeIdeogram4Transformer(&cfg.UnconditionalTransformer)
	if cfg.ModelIndex.ClassName != "" && cfg.ModelIndex.ClassName != "Ideogram4Pipeline" {
		return fmt.Errorf("unsupported Ideogram4 pipeline %q", cfg.ModelIndex.ClassName)
	}
	if err := validateIdeogram4Transformer("transformer", cfg.Transformer); err != nil {
		return err
	}
	if err := validateIdeogram4Transformer("unconditional_transformer", cfg.UnconditionalTransformer); err != nil {
		return err
	}
	if cfg.Transformer.EmbDim != cfg.UnconditionalTransformer.EmbDim || cfg.Transformer.NumLayers != cfg.UnconditionalTransformer.NumLayers || cfg.Transformer.NumHeads != cfg.UnconditionalTransformer.NumHeads {
		return fmt.Errorf("conditional/unconditional Ideogram4 transformer shapes differ")
	}
	normalizeIdeogram4TextEncoder(&cfg.TextEncoder)
	if cfg.TextEncoder.ModelType == "" && len(cfg.TextEncoder.Architectures) == 0 {
		return fmt.Errorf("invalid Ideogram4 text encoder config")
	}
	if cfg.TextEncoder.HiddenSize > 0 {
		want := cfg.TextEncoder.HiddenSize * len(Ideogram4Qwen3VLActivationLayers)
		if cfg.Transformer.LLMFeaturesDim > 0 && cfg.Transformer.LLMFeaturesDim != want {
			return fmt.Errorf("Ideogram4 llm_features_dim=%d want %d from text hidden/layers", cfg.Transformer.LLMFeaturesDim, want)
		}
	}
	if cfg.VAE.ClassName != "" && cfg.VAE.ClassName != "AutoencoderKL" && cfg.VAE.ClassName != "AutoencoderKLFlux2" {
		return fmt.Errorf("unsupported Ideogram4 VAE %q", cfg.VAE.ClassName)
	}
	return nil
}

func normalizeIdeogram4Transformer(cfg *Ideogram4TransformerConfig) {
	if cfg == nil {
		return
	}
	if cfg.NumHeads == 0 {
		cfg.NumHeads = cfg.NumAttentionHeads
	}
	if cfg.EmbDim == 0 && cfg.NumHeads > 0 && cfg.AttentionHeadDim > 0 {
		cfg.EmbDim = cfg.NumHeads * cfg.AttentionHeadDim
	}
}

func normalizeIdeogram4TextEncoder(cfg *Ideogram4TextEncoderConfig) {
	if cfg == nil || cfg.TextConfig == nil {
		return
	}
	if cfg.HiddenSize == 0 {
		cfg.HiddenSize = cfg.TextConfig.HiddenSize
	}
	if cfg.NumHiddenLayers == 0 {
		cfg.NumHiddenLayers = cfg.TextConfig.NumHiddenLayers
	}
	if cfg.NumAttentionHeads == 0 {
		cfg.NumAttentionHeads = cfg.TextConfig.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.TextConfig.NumKeyValueHeads
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = cfg.TextConfig.VocabSize
	}
}

func validateIdeogram4Transformer(name string, cfg Ideogram4TransformerConfig) error {
	if cfg.ClassName != "" && cfg.ClassName != "Ideogram4Transformer" && cfg.ClassName != "Ideogram4Transformer2DModel" {
		return fmt.Errorf("unsupported Ideogram4 %s class %q", name, cfg.ClassName)
	}
	if cfg.EmbDim <= 0 || cfg.NumLayers <= 0 || cfg.NumHeads <= 0 || cfg.IntermediateSize <= 0 || cfg.InChannels <= 0 {
		return fmt.Errorf("invalid Ideogram4 %s dimensions", name)
	}
	if cfg.EmbDim%cfg.NumHeads != 0 {
		return fmt.Errorf("Ideogram4 %s emb_dim %d not divisible by heads %d", name, cfg.EmbDim, cfg.NumHeads)
	}
	if cfg.LLMFeaturesDim <= 0 {
		return fmt.Errorf("invalid Ideogram4 %s llm_features_dim", name)
	}
	return nil
}

func SummarizeIdeogram4Config(cfg Ideogram4Config) Ideogram4Summary {
	normalizeIdeogram4Transformer(&cfg.Transformer)
	normalizeIdeogram4Transformer(&cfg.UnconditionalTransformer)
	normalizeIdeogram4TextEncoder(&cfg.TextEncoder)
	headDim := 0
	if cfg.Transformer.NumHeads > 0 {
		headDim = cfg.Transformer.EmbDim / cfg.Transformer.NumHeads
	}
	return Ideogram4Summary{
		Pipeline:                 cfg.ModelIndex.ClassName,
		Transformer:              lastIdeogram4Component(cfg.ModelIndex.Transformer),
		UnconditionalTransformer: lastIdeogram4Component(cfg.ModelIndex.UnconditionalTransformer),
		TextEncoder:              lastIdeogram4Component(cfg.ModelIndex.TextEncoder),
		Tokenizer:                lastIdeogram4Component(cfg.ModelIndex.Tokenizer),
		Scheduler:                lastIdeogram4Component(cfg.ModelIndex.Scheduler),
		VAE:                      lastIdeogram4Component(cfg.ModelIndex.VAE),
		EmbDim:                   cfg.Transformer.EmbDim,
		Layers:                   cfg.Transformer.NumLayers,
		Heads:                    cfg.Transformer.NumHeads,
		HeadDim:                  headDim,
		IntermediateSize:         cfg.Transformer.IntermediateSize,
		AdaLNDim:                 cfg.Transformer.AdaLNDim,
		InChannels:               cfg.Transformer.InChannels,
		LLMFeaturesDim:           cfg.Transformer.LLMFeaturesDim,
		MRoPESection:             append([]int(nil), cfg.Transformer.MRoPESection...),
		ActivationLayers:         append([]int(nil), Ideogram4Qwen3VLActivationLayers...),
		TextHidden:               cfg.TextEncoder.HiddenSize,
		TextLayers:               cfg.TextEncoder.NumHiddenLayers,
		VocabSize:                cfg.TextEncoder.VocabSize,
		RuntimeReady:             false,
		RuntimeNote:              "inspection only: Qwen3-VL hidden-state conditioning, FP8/NF4 Ideogram4 DiT, asymmetric CFG scheduler, and VAE decode are not implemented in native Go/SIMD yet",
	}
}

func lastIdeogram4Component(v []string) string {
	if len(v) == 0 {
		return ""
	}
	return v[len(v)-1]
}
