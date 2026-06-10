package diffusiongemma

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// Shape summarizes the DiffusionGemma checkpoint dimensions relevant to a
// future native block-diffusion runtime.
type Shape struct {
	Architecture        string   `json:"architecture"`
	ModelType           string   `json:"model_type"`
	DType               string   `json:"dtype"`
	CanvasLength        int      `json:"canvas_length"`
	TextHiddenSize      int      `json:"text_hidden_size"`
	TextLayers          int      `json:"text_layers"`
	TextHeads           int      `json:"text_heads"`
	TextKVHeads         int      `json:"text_kv_heads"`
	TextGlobalKVHeads   int      `json:"text_global_kv_heads"`
	TextHeadDim         int      `json:"text_head_dim"`
	VocabSize           int      `json:"vocab_size"`
	SlidingWindow       int      `json:"sliding_window"`
	LayerTypes          []string `json:"layer_types"`
	NumExperts          int      `json:"num_experts"`
	TopKExperts         int      `json:"top_k_experts"`
	MoEIntermediateSize int      `json:"moe_intermediate_size"`
	VisionHiddenSize    int      `json:"vision_hidden_size"`
	VisionLayers        int      `json:"vision_layers"`
	VisionHeads         int      `json:"vision_heads"`
	VisionSoftTokens    int      `json:"vision_soft_tokens"`
	PatchSize           int      `json:"patch_size"`
	BOITokenID          int      `json:"boi_token_id"`
	EOITokenID          int      `json:"eoi_token_id"`
	ImageTokenID        int      `json:"image_token_id"`
	RuntimeReady        bool     `json:"runtime_ready"`
	RuntimeNote         string   `json:"runtime_note"`
}

type GenerationDefaults struct {
	MaxNewTokens        int     `json:"max_new_tokens"`
	MaxDenoisingSteps   int     `json:"max_denoising_steps"`
	TMin                float64 `json:"t_min"`
	TMax                float64 `json:"t_max"`
	StabilityThreshold  int     `json:"stability_threshold"`
	ConfidenceThreshold float64 `json:"confidence_threshold"`
	PadTokenID          int     `json:"pad_token_id"`
	EOSTokenID          []int   `json:"eos_token_id"`
	EntropyBound        float64 `json:"entropy_bound"`
}

func ShapeFromConfig(cfg loaderconfig.DiffusionGemmaConfig) (Shape, error) {
	if err := loaderconfig.ValidateDiffusionGemmaConfig(cfg); err != nil {
		return Shape{}, err
	}
	arch := ""
	if len(cfg.Architectures) > 0 {
		arch = cfg.Architectures[0]
	}
	return Shape{
		Architecture:        arch,
		ModelType:           cfg.ModelType,
		DType:               cfg.DType,
		CanvasLength:        cfg.CanvasLength,
		TextHiddenSize:      cfg.TextConfig.HiddenSize,
		TextLayers:          cfg.TextConfig.NumHiddenLayers,
		TextHeads:           cfg.TextConfig.NumAttentionHeads,
		TextKVHeads:         cfg.TextConfig.NumKeyValueHeads,
		TextGlobalKVHeads:   cfg.TextConfig.NumGlobalKeyValueHeads,
		TextHeadDim:         cfg.TextConfig.HeadDim,
		VocabSize:           cfg.TextConfig.VocabSize,
		SlidingWindow:       cfg.TextConfig.SlidingWindow,
		LayerTypes:          append([]string(nil), cfg.TextConfig.LayerTypes...),
		NumExperts:          cfg.TextConfig.NumExperts,
		TopKExperts:         cfg.TextConfig.TopKExperts,
		MoEIntermediateSize: cfg.TextConfig.MoEIntermediateSize,
		VisionHiddenSize:    cfg.VisionConfig.HiddenSize,
		VisionLayers:        cfg.VisionConfig.NumHiddenLayers,
		VisionHeads:         cfg.VisionConfig.NumAttentionHeads,
		VisionSoftTokens:    cfg.VisionSoftTokensPerImage,
		PatchSize:           cfg.VisionConfig.PatchSize,
		BOITokenID:          cfg.BOITokenID,
		EOITokenID:          cfg.EOITokenID,
		ImageTokenID:        cfg.ImageTokenID,
		RuntimeReady:        false,
		RuntimeNote:         "inspection only: native block-diffusion sampler and DiffusionGemma forward runtime are not implemented yet",
	}, nil
}

func GenerationDefaultsFromConfig(cfg loaderconfig.DiffusionGemmaGenerationConfig) GenerationDefaults {
	return GenerationDefaults{MaxNewTokens: cfg.MaxNewTokens, MaxDenoisingSteps: cfg.MaxDenoisingSteps, TMin: cfg.TMin, TMax: cfg.TMax, StabilityThreshold: cfg.StabilityThreshold, ConfidenceThreshold: cfg.ConfidenceThreshold, PadTokenID: cfg.PadTokenID, EOSTokenID: append([]int(nil), cfg.EOSTokenID...), EntropyBound: cfg.SamplerConfig.EntropyBound}
}

func RequireRuntimeReady(s Shape) error {
	if s.RuntimeReady {
		return nil
	}
	if s.RuntimeNote != "" {
		return fmt.Errorf("%s", s.RuntimeNote)
	}
	return fmt.Errorf("DiffusionGemma runtime is not implemented")
}
