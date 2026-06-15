package config

import (
	"fmt"
	"path/filepath"
)

// DiffusionGemmaConfig captures the published Hugging Face config shape for
// google/diffusiongemma-26B-A4B-it. It is deliberately metadata-only: runtime
// block-diffusion generation is scaffolded separately from existing AR decoders.
type DiffusionGemmaConfig struct {
	Architectures            []string                   `json:"architectures"`
	ModelType                string                     `json:"model_type"`
	DType                    string                     `json:"dtype"`
	CanvasLength             int                        `json:"canvas_length"`
	BOITokenID               int                        `json:"boi_token_id"`
	EOITokenID               int                        `json:"eoi_token_id"`
	ImageTokenID             int                        `json:"image_token_id"`
	TieWordEmbeddings        bool                       `json:"tie_word_embeddings"`
	VisionSoftTokensPerImage int                        `json:"vision_soft_tokens_per_image"`
	TextConfig               DiffusionGemmaTextConfig   `json:"text_config"`
	VisionConfig             DiffusionGemmaVisionConfig `json:"vision_config"`
	TransformersVersion      string                     `json:"transformers_version"`
}

type DiffusionGemmaTextConfig struct {
	ModelType                 string         `json:"model_type"`
	DType                     string         `json:"dtype"`
	HiddenSize                int            `json:"hidden_size"`
	NumHiddenLayers           int            `json:"num_hidden_layers"`
	NumAttentionHeads         int            `json:"num_attention_heads"`
	NumKeyValueHeads          int            `json:"num_key_value_heads"`
	NumGlobalKeyValueHeads    int            `json:"num_global_key_value_heads"`
	HeadDim                   int            `json:"head_dim"`
	GlobalHeadDim             int            `json:"global_head_dim"`
	IntermediateSize          int            `json:"intermediate_size"`
	MoEIntermediateSize       int            `json:"moe_intermediate_size"`
	NumExperts                int            `json:"num_experts"`
	TopKExperts               int            `json:"top_k_experts"`
	VocabSize                 int            `json:"vocab_size"`
	MaxPositionEmbeddings     int            `json:"max_position_embeddings"`
	SlidingWindow             int            `json:"sliding_window"`
	LayerTypes                []string       `json:"layer_types"`
	UseBidirectionalAttention string         `json:"use_bidirectional_attention"`
	HiddenActivation          string         `json:"hidden_activation"`
	RMSNormEps                float64        `json:"rms_norm_eps"`
	BOSTokenID                int            `json:"bos_token_id"`
	EOSTokenID                int            `json:"eos_token_id"`
	PadTokenID                int            `json:"pad_token_id"`
	FinalLogitSoftcapping     float64        `json:"final_logit_softcapping"`
	RopeParameters            map[string]any `json:"rope_parameters"`
	TieWordEmbeddings         bool           `json:"tie_word_embeddings"`
}

type DiffusionGemmaVisionConfig struct {
	ModelType             string         `json:"model_type"`
	DType                 string         `json:"dtype"`
	HiddenSize            int            `json:"hidden_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	HeadDim               int            `json:"head_dim"`
	GlobalHeadDim         int            `json:"global_head_dim"`
	IntermediateSize      int            `json:"intermediate_size"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	PatchSize             int            `json:"patch_size"`
	PositionEmbeddingSize int            `json:"position_embedding_size"`
	DefaultOutputLength   int            `json:"default_output_length"`
	PoolingKernelSize     int            `json:"pooling_kernel_size"`
	HiddenActivation      string         `json:"hidden_activation"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	RopeParameters        map[string]any `json:"rope_parameters"`
	Standardize           bool           `json:"standardize"`
	UseClippedLinears     bool           `json:"use_clipped_linears"`
}

type DiffusionGemmaGenerationConfig struct {
	MaxNewTokens        int                         `json:"max_new_tokens"`
	MaxDenoisingSteps   int                         `json:"max_denoising_steps"`
	TMin                float64                     `json:"t_min"`
	TMax                float64                     `json:"t_max"`
	StabilityThreshold  int                         `json:"stability_threshold"`
	ConfidenceThreshold float64                     `json:"confidence_threshold"`
	PadTokenID          int                         `json:"pad_token_id"`
	EOSTokenID          []int                       `json:"eos_token_id"`
	SamplerConfig       DiffusionGemmaSamplerConfig `json:"sampler_config"`
}

type DiffusionGemmaSamplerConfig struct {
	ClassName    string  `json:"_cls_name"`
	EntropyBound float64 `json:"entropy_bound"`
}

func ReadDiffusionGemmaConfig(dir string) (DiffusionGemmaConfig, error) {
	var cfg DiffusionGemmaConfig
	_, err := ReadJSON(filepath.Join(dir, "config.json"), &cfg)
	if err != nil {
		return cfg, err
	}
	return cfg, ValidateDiffusionGemmaConfig(cfg)
}

func ReadDiffusionGemmaGenerationConfig(dir string) (DiffusionGemmaGenerationConfig, bool, error) {
	// Use negative sentinels for controls where zero is an explicit, valid
	// DiffusionGemma setting. Missing JSON fields keep the sentinel so the model
	// package can fall back to llama.cpp defaults; explicit 0 remains 0.
	cfg := DiffusionGemmaGenerationConfig{
		StabilityThreshold:  -1,
		ConfidenceThreshold: -1,
		SamplerConfig: DiffusionGemmaSamplerConfig{
			EntropyBound: -1,
		},
	}
	ok, err := ReadOptionalJSON(filepath.Join(dir, "generation_config.json"), &cfg)
	return cfg, ok, err
}

func ValidateDiffusionGemmaConfig(cfg DiffusionGemmaConfig) error {
	if cfg.ModelType != "diffusion_gemma" {
		return fmt.Errorf("unsupported DiffusionGemma model_type %q", cfg.ModelType)
	}
	if len(cfg.Architectures) == 0 || cfg.Architectures[0] != "DiffusionGemmaForBlockDiffusion" {
		return fmt.Errorf("unsupported DiffusionGemma architecture %v", cfg.Architectures)
	}
	if cfg.CanvasLength <= 0 || cfg.TextConfig.HiddenSize <= 0 || cfg.TextConfig.NumHiddenLayers <= 0 || cfg.TextConfig.VocabSize <= 0 {
		return fmt.Errorf("invalid DiffusionGemma text dimensions")
	}
	if cfg.TextConfig.NumExperts <= 0 || cfg.TextConfig.TopKExperts <= 0 {
		return fmt.Errorf("invalid DiffusionGemma MoE dimensions")
	}
	if cfg.VisionConfig.HiddenSize < 0 || cfg.VisionSoftTokensPerImage < 0 {
		return fmt.Errorf("invalid DiffusionGemma vision dimensions")
	}
	return nil
}
