package config

import (
	"encoding/json"
	"fmt"
	"path/filepath"
)

// MiniCPMVConfig captures the Hugging Face/OpenBMB MiniCPM-V/MiniCPM-O family
// config shape. Published checkpoints use trust_remote_code and have evolved
// from the original OmniLMM/Mistral text stack to MiniCPM/Qwen2 text backbones,
// so this struct intentionally accepts both top-level and nested text/vision
// fields.
type MiniCPMVConfig struct {
	Architectures       []string                 `json:"architectures"`
	ModelType           string                   `json:"model_type"`
	DType               string                   `json:"torch_dtype"`
	HiddenSize          int                      `json:"hidden_size"`
	NumHiddenLayers     int                      `json:"num_hidden_layers"`
	NumAttentionHeads   int                      `json:"num_attention_heads"`
	NumKeyValueHeads    int                      `json:"num_key_value_heads"`
	HeadDim             int                      `json:"head_dim"`
	IntermediateSize    int                      `json:"intermediate_size"`
	VocabSize           int                      `json:"vocab_size"`
	MaxPositionEmbeds   int                      `json:"max_position_embeddings"`
	RMSNormEps          float64                  `json:"rms_norm_eps"`
	RopeTheta           float64                  `json:"rope_theta"`
	RopeScaling         json.RawMessage          `json:"rope_scaling"`
	HiddenAct           string                   `json:"hidden_act"`
	AttentionBias       *bool                    `json:"attention_bias"`
	ScaleDepth          float64                  `json:"scale_depth"`
	ScaleEmb            float64                  `json:"scale_emb"`
	DimModelBase        int                      `json:"dim_model_base"`
	UseSlidingWindow    bool                     `json:"use_sliding_window"`
	SlidingWindow       int                      `json:"sliding_window"`
	BOSTokenID          int                      `json:"bos_token_id"`
	EOSTokenID          any                      `json:"eos_token_id"`
	PadTokenID          int                      `json:"pad_token_id"`
	TieWordEmbeddings   bool                     `json:"tie_word_embeddings"`
	MMVisionTower       string                   `json:"mm_vision_tower"`
	VisionEncoder       string                   `json:"vision_encoder"`
	DropVisionLastLayer bool                     `json:"drop_vision_last_layer"`
	UseMMProj           bool                     `json:"use_mm_proj"`
	NumQuery            int                      `json:"num_query"`
	QueryNum            int                      `json:"query_num"`
	ImageSize           int                      `json:"image_size"`
	PatchSize           int                      `json:"patch_size"`
	MaxSliceNums        int                      `json:"max_slice_nums"`
	UseImageStartEnd    *bool                    `json:"use_im_start_end"`
	ImageTokenID        int                      `json:"image_token_id"`
	ImageStartTokenID   int                      `json:"im_start_token_id"`
	ImageEndTokenID     int                      `json:"im_end_token_id"`
	SliceMode           *bool                    `json:"slice_mode"`
	SliceConfig         MiniCPMVSliceConfig      `json:"slice_config"`
	TextConfig          *MiniCPMVTextConfig      `json:"text_config"`
	VisionConfig        *MiniCPMVVisionConfig    `json:"vision_config"`
	AudioConfig         *MiniCPMOAudioConfig     `json:"audio_config"`
	AudioPoolStep       int                      `json:"audio_pool_step"`
	ResamplerConfig     *MiniCPMVResamplerConfig `json:"resampler_config"`
	TransformersVersion string                   `json:"transformers_version"`
}

type MiniCPMVTextConfig struct {
	ModelType         string          `json:"model_type"`
	HiddenSize        int             `json:"hidden_size"`
	NumHiddenLayers   int             `json:"num_hidden_layers"`
	NumAttentionHeads int             `json:"num_attention_heads"`
	NumKeyValueHeads  int             `json:"num_key_value_heads"`
	HeadDim           int             `json:"head_dim"`
	IntermediateSize  int             `json:"intermediate_size"`
	VocabSize         int             `json:"vocab_size"`
	MaxPositionEmbeds int             `json:"max_position_embeddings"`
	RMSNormEps        float64         `json:"rms_norm_eps"`
	RopeTheta         float64         `json:"rope_theta"`
	RopeScaling       json.RawMessage `json:"rope_scaling"`
	HiddenAct         string          `json:"hidden_act"`
	AttentionBias     *bool           `json:"attention_bias"`
	TieWordEmbeddings *bool           `json:"tie_word_embeddings"`
	ScaleDepth        float64         `json:"scale_depth"`
	ScaleEmb          float64         `json:"scale_emb"`
	DimModelBase      int             `json:"dim_model_base"`
	UseSlidingWindow  bool            `json:"use_sliding_window"`
	SlidingWindow     int             `json:"sliding_window"`
}

type MiniCPMVVisionConfig struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	NumChannels       int     `json:"num_channels"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	IntermediateSize  int     `json:"intermediate_size"`
	ProjectionDim     int     `json:"projection_dim"`
	HiddenAct         string  `json:"hidden_act"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	AttentionDropout  float64 `json:"attention_dropout"`
	DType             string  `json:"torch_dtype"`
}

type MiniCPMOAudioConfig struct {
	ModelType          string  `json:"model_type"`
	HiddenSize         int     `json:"hidden_size"`
	DModel             int     `json:"d_model"`
	NumHiddenLayers    int     `json:"num_hidden_layers"`
	EncoderLayers      int     `json:"encoder_layers"`
	NumAttentionHeads  int     `json:"num_attention_heads"`
	EncoderHeads       int     `json:"encoder_attention_heads"`
	IntermediateSize   int     `json:"intermediate_size"`
	EncoderFFNDim      int     `json:"encoder_ffn_dim"`
	FeatureSize        int     `json:"feature_size"`
	NumMelBins         int     `json:"num_mel_bins"`
	SamplingRate       int     `json:"sampling_rate"`
	MaxSourcePositions int     `json:"max_source_positions"`
	ActivationFunction string  `json:"activation_function"`
	LayerNormEps       float64 `json:"layer_norm_eps"`
	Dropout            float64 `json:"dropout"`
	ActivationDropout  float64 `json:"activation_dropout"`
	AttentionDropout   float64 `json:"attention_dropout"`
}

type MiniCPMVResamplerConfig struct {
	NumQuery int `json:"num_query"`
	NumHeads int `json:"num_heads"`
	EmbedDim int `json:"embed_dim"`
	KVDim    int `json:"kv_dim"`
	GridSize int `json:"grid_size"`
}

type MiniCPMVSliceConfig struct {
	MaxSliceNums    int `json:"max_slice_nums"`
	ScaleResolution int `json:"scale_resolution"`
	PatchSize       int `json:"patch_size"`
}

// MiniCPMVSummary is the normalized readiness surface consumed by model/cmd
// packages without having to understand every upstream config variant.
type MiniCPMVSummary struct {
	Architecture            string `json:"architecture"`
	ModelType               string `json:"model_type"`
	TextModelType           string `json:"text_model_type"`
	VisionModelType         string `json:"vision_model_type"`
	HiddenSize              int    `json:"hidden_size"`
	Layers                  int    `json:"layers"`
	Heads                   int    `json:"heads"`
	KVHeads                 int    `json:"kv_heads"`
	HeadDim                 int    `json:"head_dim"`
	IntermediateSize        int    `json:"intermediate_size"`
	VocabSize               int    `json:"vocab_size"`
	ImageSize               int    `json:"image_size"`
	PatchSize               int    `json:"patch_size"`
	VisionHiddenSize        int    `json:"vision_hidden_size"`
	VisionIntermediate      int    `json:"vision_intermediate_size"`
	VisionLayers            int    `json:"vision_layers"`
	VisionHeads             int    `json:"vision_heads"`
	AudioModelType          string `json:"audio_model_type,omitempty"`
	AudioHiddenSize         int    `json:"audio_hidden_size,omitempty"`
	AudioIntermediateSize   int    `json:"audio_intermediate_size,omitempty"`
	AudioLayers             int    `json:"audio_layers,omitempty"`
	AudioHeads              int    `json:"audio_heads,omitempty"`
	AudioFeatureSize        int    `json:"audio_feature_size,omitempty"`
	AudioMelBins            int    `json:"audio_mel_bins,omitempty"`
	AudioSamplingRate       int    `json:"audio_sampling_rate,omitempty"`
	AudioMaxSourcePositions int    `json:"audio_max_source_positions,omitempty"`
	AudioPoolStep           int    `json:"audio_pool_step,omitempty"`
	NumQuery                int    `json:"num_query"`
	ResamplerGrid           int    `json:"resampler_grid"`
	ResamplerHeads          int    `json:"resampler_heads"`
	UseImageStartEnd        bool   `json:"use_image_start_end"`
	ImageTokenID            int    `json:"image_token_id"`
	ImageStartTokenID       int    `json:"image_start_token_id"`
	ImageEndTokenID         int    `json:"image_end_token_id"`
	SliceMode               bool   `json:"slice_mode"`
	MaxSliceNums            int    `json:"max_slice_nums,omitempty"`
	ScaleResolution         int    `json:"scale_resolution,omitempty"`
	SlicePatchSize          int    `json:"slice_patch_size,omitempty"`
	RuntimeReady            bool   `json:"runtime_ready"`
	RuntimeNote             string `json:"runtime_note"`
}

func ReadMiniCPMVConfig(dir string) (MiniCPMVConfig, error) {
	var cfg MiniCPMVConfig
	_, err := ReadJSON(filepath.Join(dir, "config.json"), &cfg)
	if err != nil {
		return cfg, err
	}
	return cfg, ValidateMiniCPMVConfig(cfg)
}

func ValidateMiniCPMVConfig(cfg MiniCPMVConfig) error {
	s := cfg.MiniCPMVSummary()
	if !isMiniCPMVModelType(s.ModelType) && !hasMiniCPMVArchitecture(cfg.Architectures) {
		return fmt.Errorf("unsupported MiniCPM-V/O model_type=%q architectures=%v", cfg.ModelType, cfg.Architectures)
	}
	if s.HiddenSize <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.VocabSize <= 0 {
		return fmt.Errorf("invalid MiniCPM-V text dimensions: hidden=%d layers=%d heads=%d vocab=%d", s.HiddenSize, s.Layers, s.Heads, s.VocabSize)
	}
	if s.NumQuery <= 0 {
		return fmt.Errorf("invalid MiniCPM-V num_query %d", s.NumQuery)
	}
	if s.ResamplerGrid*s.ResamplerGrid != s.NumQuery {
		return fmt.Errorf("MiniCPM-V num_query=%d is not a square resampler grid", s.NumQuery)
	}
	if s.ImageSize < 0 || s.PatchSize < 0 || s.VisionHiddenSize < 0 {
		return fmt.Errorf("invalid MiniCPM-V vision dimensions")
	}
	return nil
}

func (cfg MiniCPMVConfig) MiniCPMVSummary() MiniCPMVSummary {
	s := MiniCPMVSummary{ModelType: cfg.ModelType, RuntimeReady: false}
	if len(cfg.Architectures) > 0 {
		s.Architecture = cfg.Architectures[0]
	}
	s.HiddenSize = firstPositive(cfg.HiddenSize, textInt(cfg.TextConfig, "hidden"))
	s.Layers = firstPositive(cfg.NumHiddenLayers, textInt(cfg.TextConfig, "layers"))
	s.Heads = firstPositive(cfg.NumAttentionHeads, textInt(cfg.TextConfig, "heads"))
	s.KVHeads = firstPositive(cfg.NumKeyValueHeads, textInt(cfg.TextConfig, "kvheads"), s.Heads)
	s.HeadDim = firstPositive(cfg.HeadDim, textInt(cfg.TextConfig, "headdim"))
	if s.HeadDim == 0 && s.Heads > 0 {
		s.HeadDim = s.HiddenSize / s.Heads
	}
	s.IntermediateSize = firstPositive(cfg.IntermediateSize, textInt(cfg.TextConfig, "intermediate"))
	s.VocabSize = firstPositive(cfg.VocabSize, textInt(cfg.TextConfig, "vocab"))
	if cfg.TextConfig != nil {
		s.TextModelType = cfg.TextConfig.ModelType
	}
	if s.TextModelType == "" {
		switch cfg.ModelType {
		case "omnilmm":
			s.TextModelType = "mistral"
		case "minicpmo", "minicpm_o", "minicpm-o":
			s.TextModelType = "qwen2"
		case "minicpmv", "minicpm_v", "minicpm-v":
			if cfg.ScaleEmb != 0 || cfg.ScaleDepth != 0 || cfg.DimModelBase != 0 {
				s.TextModelType = "minicpm"
			} else {
				s.TextModelType = "qwen2"
			}
		}
	}
	if cfg.VisionConfig != nil {
		s.VisionModelType = cfg.VisionConfig.ModelType
		s.VisionHiddenSize = cfg.VisionConfig.HiddenSize
		s.VisionIntermediate = cfg.VisionConfig.IntermediateSize
		s.VisionLayers = cfg.VisionConfig.NumHiddenLayers
		s.VisionHeads = cfg.VisionConfig.NumAttentionHeads
		s.ImageSize = firstPositive(cfg.ImageSize, cfg.VisionConfig.ImageSize)
		s.PatchSize = firstPositive(cfg.PatchSize, cfg.VisionConfig.PatchSize)
	} else {
		s.ImageSize = cfg.ImageSize
		s.PatchSize = cfg.PatchSize
		if cfg.VisionEncoder != "" {
			s.VisionModelType = cfg.VisionEncoder
		}
		// The published v2.0 checkpoint uses this exact timm SigLIP So400m
		// preset. Record its architecture dimensions even though they are not
		// repeated in config.json.
		if cfg.VisionEncoder == "vit_so400m_patch14_siglip_384.webli" {
			s.VisionHiddenSize = 1152
			s.VisionIntermediate = 4304
			s.VisionLayers = 27
			if cfg.DropVisionLastLayer {
				s.VisionLayers--
			}
			s.VisionHeads = 16
		}
	}
	if cfg.AudioConfig != nil {
		audio := cfg.AudioConfig
		s.AudioModelType = audio.ModelType
		s.AudioHiddenSize = firstPositive(audio.HiddenSize, audio.DModel)
		s.AudioIntermediateSize = firstPositive(audio.IntermediateSize, audio.EncoderFFNDim)
		s.AudioLayers = firstPositive(audio.NumHiddenLayers, audio.EncoderLayers)
		s.AudioHeads = firstPositive(audio.NumAttentionHeads, audio.EncoderHeads)
		s.AudioFeatureSize = firstPositive(audio.FeatureSize, audio.NumMelBins)
		s.AudioMelBins = audio.NumMelBins
		s.AudioSamplingRate = audio.SamplingRate
		s.AudioMaxSourcePositions = audio.MaxSourcePositions
		if audio.ModelType == "whisper" {
			// These are WhisperConfig/WhisperFeatureExtractor defaults. The
			// published MiniCPM-O 2.6 nested config relies on them rather than
			// serializing each value.
			s.AudioFeatureSize = firstPositive(s.AudioFeatureSize, 80)
			s.AudioMelBins = firstPositive(s.AudioMelBins, 80)
			s.AudioSamplingRate = firstPositive(s.AudioSamplingRate, 16000)
			s.AudioMaxSourcePositions = firstPositive(s.AudioMaxSourcePositions, 1500)
		}
		s.AudioPoolStep = firstPositive(cfg.AudioPoolStep, 2)
	}
	s.NumQuery = firstPositive(cfg.NumQuery, cfg.QueryNum, resamplerInt(cfg.ResamplerConfig, "numquery"))
	s.ResamplerGrid = firstPositive(resamplerInt(cfg.ResamplerConfig, "grid"), sqrtInt(s.NumQuery))
	s.ResamplerHeads = firstPositive(resamplerInt(cfg.ResamplerConfig, "heads"), s.HiddenSize/128)
	if cfg.UseImageStartEnd == nil {
		s.UseImageStartEnd = true
	} else {
		s.UseImageStartEnd = *cfg.UseImageStartEnd
	}
	s.ImageTokenID = cfg.ImageTokenID
	s.ImageStartTokenID = cfg.ImageStartTokenID
	s.ImageEndTokenID = cfg.ImageEndTokenID
	if cfg.SliceMode != nil {
		s.SliceMode = *cfg.SliceMode
	}
	s.MaxSliceNums = firstPositive(cfg.SliceConfig.MaxSliceNums, cfg.MaxSliceNums)
	s.ScaleResolution = cfg.SliceConfig.ScaleResolution
	s.SlicePatchSize = cfg.SliceConfig.PatchSize
	s.RuntimeNote = "CPU text, nested/fused-QKV SigLIP, resampler, and MiniCPM-O audio slices implemented; released-model parity and end-to-end generation pending"
	return s
}

func isMiniCPMVModelType(t string) bool {
	switch t {
	case "minicpmv", "minicpm_v", "minicpm-v", "minicpmo", "minicpm_o", "minicpm-o", "omnilmm":
		return true
	default:
		return false
	}
}

func hasMiniCPMVArchitecture(arch []string) bool {
	for _, a := range arch {
		switch a {
		case "MiniCPMV", "MiniCPMVForCausalLM", "MiniCPMVChatModel", "MiniCPMO", "MiniCPMOForCausalLM", "MiniCPMOChatModel", "OmniLMMForCausalLM":
			return true
		}
	}
	return false
}

func firstPositive(v int, rest ...int) int {
	if v > 0 {
		return v
	}
	for _, x := range rest {
		if x > 0 {
			return x
		}
	}
	return 0
}

func textInt(c *MiniCPMVTextConfig, k string) int {
	if c == nil {
		return 0
	}
	switch k {
	case "hidden":
		return c.HiddenSize
	case "layers":
		return c.NumHiddenLayers
	case "heads":
		return c.NumAttentionHeads
	case "kvheads":
		return c.NumKeyValueHeads
	case "headdim":
		return c.HeadDim
	case "intermediate":
		return c.IntermediateSize
	case "vocab":
		return c.VocabSize
	default:
		return 0
	}
}

func resamplerInt(c *MiniCPMVResamplerConfig, k string) int {
	if c == nil {
		return 0
	}
	switch k {
	case "numquery":
		return c.NumQuery
	case "heads":
		return c.NumHeads
	case "grid":
		return c.GridSize
	default:
		return 0
	}
}

func sqrtInt(n int) int {
	if n <= 0 {
		return 0
	}
	for i := 1; i*i <= n; i++ {
		if i*i == n {
			return i
		}
	}
	return 0
}
