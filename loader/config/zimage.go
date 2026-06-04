package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ZImageModelIndex describes the Diffusers model_index.json published by
// Tongyi-MAI/Z-Image-Turbo. It is intentionally small: generation support needs
// separate DiT, VAE, scheduler, and text-encoder runtimes, but this gives the
// loader a stable readiness surface.
type ZImageModelIndex struct {
	ClassName        string   `json:"_class_name"`
	DiffusersVersion string   `json:"_diffusers_version"`
	Scheduler        []string `json:"scheduler"`
	TextEncoder      []string `json:"text_encoder"`
	Tokenizer        []string `json:"tokenizer"`
	Transformer      []string `json:"transformer"`
	VAE              []string `json:"vae"`
}

type ZImageTransformerConfig struct {
	ClassName      string `json:"_class_name"`
	Dim            int    `json:"dim"`
	InChannels     int    `json:"in_channels"`
	NHeads         int    `json:"n_heads"`
	NKVHeads       int    `json:"n_kv_heads"`
	NLayers        int    `json:"n_layers"`
	NRefinerLayers int    `json:"n_refiner_layers"`
	CapFeatDim     int    `json:"cap_feat_dim"`
	AxesDims       []int  `json:"axes_dims"`
	AxesLens       []int  `json:"axes_lens"`
	PatchSize      int    `json:"patch_size"`
	OutChannels    int    `json:"out_channels"`
}

type ZImageVAEConfig struct {
	ClassName      string  `json:"_class_name"`
	InChannels     int     `json:"in_channels"`
	OutChannels    int     `json:"out_channels"`
	LatentChannels int     `json:"latent_channels"`
	SampleSize     int     `json:"sample_size"`
	ScalingFactor  float64 `json:"scaling_factor"`
}

type ZImageTextEncoderConfig struct {
	ModelType         string   `json:"model_type"`
	Architectures     []string `json:"architectures"`
	HiddenSize        int      `json:"hidden_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	VocabSize         int      `json:"vocab_size"`
}

type ZImageSchedulerConfig struct {
	ClassName         string  `json:"_class_name"`
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	Shift             float64 `json:"shift"`
}

type ZImageConfig struct {
	Root        string                  `json:"root"`
	ModelIndex  ZImageModelIndex        `json:"model_index"`
	Transformer ZImageTransformerConfig `json:"transformer"`
	VAE         ZImageVAEConfig         `json:"vae"`
	TextEncoder ZImageTextEncoderConfig `json:"text_encoder"`
	Scheduler   ZImageSchedulerConfig   `json:"scheduler"`
}

type ZImageSummary struct {
	Pipeline      string `json:"pipeline"`
	Transformer   string `json:"transformer"`
	TextEncoder   string `json:"text_encoder"`
	Tokenizer     string `json:"tokenizer"`
	Scheduler     string `json:"scheduler"`
	VAE           string `json:"vae"`
	Dim           int    `json:"dim"`
	Layers        int    `json:"layers"`
	RefinerLayers int    `json:"refiner_layers"`
	Heads         int    `json:"heads"`
	KVHeads       int    `json:"kv_heads"`
	InChannels    int    `json:"in_channels"`
	CapFeatDim    int    `json:"cap_feat_dim"`
	AxesDims      []int  `json:"axes_dims"`
	AxesLens      []int  `json:"axes_lens"`
	TextHidden    int    `json:"text_hidden"`
	TextLayers    int    `json:"text_layers"`
	VocabSize     int    `json:"vocab_size"`
	RuntimeReady  bool   `json:"runtime_ready"`
	RuntimeNote   string `json:"runtime_note"`
}

func ReadZImageConfig(root string) (ZImageConfig, error) {
	var cfg ZImageConfig
	cfg.Root = root
	if err := readZImageJSON(filepath.Join(root, "model_index.json"), &cfg.ModelIndex); err != nil {
		return cfg, err
	}
	if err := readZImageJSON(filepath.Join(root, "transformer", "config.json"), &cfg.Transformer); err != nil {
		return cfg, err
	}
	if err := readZImageJSON(filepath.Join(root, "vae", "config.json"), &cfg.VAE); err != nil {
		return cfg, err
	}
	if err := readZImageJSON(filepath.Join(root, "text_encoder", "config.json"), &cfg.TextEncoder); err != nil {
		return cfg, err
	}
	if err := readZImageJSON(filepath.Join(root, "scheduler", "scheduler_config.json"), &cfg.Scheduler); err != nil {
		return cfg, err
	}
	return cfg, ValidateZImageConfig(cfg)
}

func readZImageJSON(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func ValidateZImageConfig(cfg ZImageConfig) error {
	if cfg.ModelIndex.ClassName != "ZImagePipeline" {
		return fmt.Errorf("unsupported Z-Image pipeline %q", cfg.ModelIndex.ClassName)
	}
	if cfg.Transformer.ClassName != "ZImageTransformer2DModel" {
		return fmt.Errorf("unsupported Z-Image transformer %q", cfg.Transformer.ClassName)
	}
	if cfg.Transformer.Dim <= 0 || cfg.Transformer.NLayers <= 0 || cfg.Transformer.NHeads <= 0 {
		return fmt.Errorf("invalid Z-Image transformer dimensions")
	}
	if cfg.Transformer.Dim%cfg.Transformer.NHeads != 0 {
		return fmt.Errorf("Z-Image dim %d not divisible by heads %d", cfg.Transformer.Dim, cfg.Transformer.NHeads)
	}
	if cfg.Transformer.InChannels <= 0 || cfg.Transformer.CapFeatDim <= 0 {
		return fmt.Errorf("invalid Z-Image channel/context dimensions")
	}
	if cfg.TextEncoder.ModelType == "" || cfg.TextEncoder.HiddenSize <= 0 || cfg.TextEncoder.NumHiddenLayers <= 0 {
		return fmt.Errorf("invalid Z-Image text encoder config")
	}
	if cfg.VAE.ClassName != "AutoencoderKL" {
		return fmt.Errorf("unsupported Z-Image VAE %q", cfg.VAE.ClassName)
	}
	return nil
}

func SummarizeZImageConfig(cfg ZImageConfig) ZImageSummary {
	return ZImageSummary{
		Pipeline:    cfg.ModelIndex.ClassName,
		Transformer: lastZImageComponent(cfg.ModelIndex.Transformer),
		TextEncoder: lastZImageComponent(cfg.ModelIndex.TextEncoder),
		Tokenizer:   lastZImageComponent(cfg.ModelIndex.Tokenizer),
		Scheduler:   lastZImageComponent(cfg.ModelIndex.Scheduler),
		VAE:         lastZImageComponent(cfg.ModelIndex.VAE),
		Dim:         cfg.Transformer.Dim, Layers: cfg.Transformer.NLayers, RefinerLayers: cfg.Transformer.NRefinerLayers,
		Heads: cfg.Transformer.NHeads, KVHeads: cfg.Transformer.NKVHeads, InChannels: cfg.Transformer.InChannels,
		CapFeatDim: cfg.Transformer.CapFeatDim, AxesDims: append([]int(nil), cfg.Transformer.AxesDims...), AxesLens: append([]int(nil), cfg.Transformer.AxesLens...),
		TextHidden: cfg.TextEncoder.HiddenSize, TextLayers: cfg.TextEncoder.NumHiddenLayers, VocabSize: cfg.TextEncoder.VocabSize,
		RuntimeReady: false,
		RuntimeNote:  "inspection only: S3-DiT flow-matching transformer, AutoencoderKL decode, and image scheduler runtime are not implemented yet",
	}
}

func lastZImageComponent(v []string) string {
	if len(v) == 0 {
		return ""
	}
	return v[len(v)-1]
}
