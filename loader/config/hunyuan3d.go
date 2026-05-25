package config

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Hunyuan3DConfig is the subset of Tencent Hunyuan3D shape-generation YAML
// needed for loader inventory and fixture planning. It intentionally models the
// shape pipeline config rather than the later texture-generation Diffusers stack.
type Hunyuan3DConfig struct {
	Model          Hunyuan3DTarget[Hunyuan3DDiTParams]         `yaml:"model"`
	VAE            Hunyuan3DTarget[Hunyuan3DShapeVAEParams]    `yaml:"vae"`
	Conditioner    Hunyuan3DTarget[Hunyuan3DConditionerParams] `yaml:"conditioner"`
	ImageProcessor Hunyuan3DTarget[map[string]any]             `yaml:"image_processor"`
	Scheduler      Hunyuan3DTarget[Hunyuan3DSchedulerParams]   `yaml:"scheduler"`
}

// Hunyuan3DTarget mirrors the upstream YAML convention of Python target strings
// plus a nested params object.
type Hunyuan3DTarget[T any] struct {
	Target string `yaml:"target"`
	Params T      `yaml:"params"`
}

type Hunyuan3DDiTParams struct {
	InChannels        int     `yaml:"in_channels"`
	ContextInDim      int     `yaml:"context_in_dim"`
	HiddenSize        int     `yaml:"hidden_size"`
	MLPRatio          float64 `yaml:"mlp_ratio"`
	NumHeads          int     `yaml:"num_heads"`
	Depth             int     `yaml:"depth"`
	DepthSingleBlocks int     `yaml:"depth_single_blocks"`
	AxesDim           []int   `yaml:"axes_dim"`
	Theta             int     `yaml:"theta"`
	QKVBias           bool    `yaml:"qkv_bias"`
	GuidanceEmbed     bool    `yaml:"guidance_embed"`
}

type Hunyuan3DShapeVAEParams struct {
	NumLatents                int     `yaml:"num_latents"`
	EmbedDim                  int     `yaml:"embed_dim"`
	Width                     int     `yaml:"width"`
	Heads                     int     `yaml:"heads"`
	NumDecoderLayers          int     `yaml:"num_decoder_layers"`
	NumEncoderLayers          int     `yaml:"num_encoder_layers"`
	PCSize                    int     `yaml:"pc_size"`
	PCSharpEdgeSize           int     `yaml:"pc_sharpedge_size"`
	PointFeats                int     `yaml:"point_feats"`
	DownsampleRatio           int     `yaml:"downsample_ratio"`
	GeoDecoderDownsampleRatio int     `yaml:"geo_decoder_downsample_ratio"`
	GeoDecoderMLPExpandRatio  int     `yaml:"geo_decoder_mlp_expand_ratio"`
	GeoDecoderLNPost          bool    `yaml:"geo_decoder_ln_post"`
	NumFreqs                  int     `yaml:"num_freqs"`
	IncludePi                 bool    `yaml:"include_pi"`
	QKVBias                   bool    `yaml:"qkv_bias"`
	QKNorm                    bool    `yaml:"qk_norm"`
	LabelType                 string  `yaml:"label_type"`
	DropPathRate              float64 `yaml:"drop_path_rate"`
	ScaleFactor               float64 `yaml:"scale_factor"`
	UseLNPost                 bool    `yaml:"use_ln_post"`
}

type Hunyuan3DConditionerParams struct {
	MainImageEncoder       Hunyuan3DImageEncoderConfig `yaml:"main_image_encoder"`
	AdditionalImageEncoder Hunyuan3DImageEncoderConfig `yaml:"additional_image_encoder"`
}

type Hunyuan3DImageEncoderConfig struct {
	Type   string         `yaml:"type"`
	Kwargs map[string]any `yaml:"kwargs"`
}

type Hunyuan3DSchedulerParams struct {
	NumTrainTimesteps  int     `yaml:"num_train_timesteps"`
	Shift              float64 `yaml:"shift"`
	UseDynamicShifting bool    `yaml:"use_dynamic_shifting"`
}

// Hunyuan3DSummary is a compact inventory of the dimensions that drive the
// first implementation/fixture milestones.
type Hunyuan3DSummary struct {
	DenoiserTarget    string
	VAETarget         string
	ConditionerTarget string
	SchedulerTarget   string
	InChannels        int
	ContextInDim      int
	HiddenSize        int
	NumHeads          int
	Depth             int
	DepthSingleBlocks int
	VAELatents        int
	VAEEmbedDim       int
	VAEWidth          int
	VAEHeads          int
	ConditionerType   string
	SchedulerSteps    int
}

func ParseHunyuan3DConfig(data []byte) (Hunyuan3DConfig, error) {
	var cfg Hunyuan3DConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse Hunyuan3D YAML: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func ReadHunyuan3DConfig(path string) (Hunyuan3DConfig, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Hunyuan3DConfig{}, nil, err
	}
	cfg, err := ParseHunyuan3DConfig(data)
	return cfg, data, err
}

func (c Hunyuan3DConfig) Validate() error {
	if c.Model.Target == "" || c.VAE.Target == "" || c.Conditioner.Target == "" || c.Scheduler.Target == "" {
		return fmt.Errorf("invalid Hunyuan3D config: missing model/vae/conditioner/scheduler target")
	}
	m := c.Model.Params
	if m.InChannels <= 0 || m.ContextInDim <= 0 || m.HiddenSize <= 0 || m.NumHeads <= 0 || m.Depth < 0 || m.DepthSingleBlocks < 0 {
		return fmt.Errorf("invalid Hunyuan3D denoiser dims: %+v", m)
	}
	if m.HiddenSize%m.NumHeads != 0 {
		return fmt.Errorf("invalid Hunyuan3D denoiser dims: hidden_size %d not divisible by num_heads %d", m.HiddenSize, m.NumHeads)
	}
	if len(m.AxesDim) > 0 {
		sum := 0
		for _, d := range m.AxesDim {
			if d <= 0 {
				return fmt.Errorf("invalid Hunyuan3D axes_dim: %v", m.AxesDim)
			}
			sum += d
		}
		if want := m.HiddenSize / m.NumHeads; sum != want {
			return fmt.Errorf("invalid Hunyuan3D axes_dim sum %d, want head dim %d", sum, want)
		}
	}
	v := c.VAE.Params
	if v.NumLatents <= 0 || v.EmbedDim <= 0 || v.Width <= 0 || v.Heads <= 0 || v.NumDecoderLayers <= 0 {
		return fmt.Errorf("invalid Hunyuan3D VAE dims: %+v", v)
	}
	if v.Width%v.Heads != 0 {
		return fmt.Errorf("invalid Hunyuan3D VAE dims: width %d not divisible by heads %d", v.Width, v.Heads)
	}
	if c.Scheduler.Params.NumTrainTimesteps <= 0 {
		return fmt.Errorf("invalid Hunyuan3D scheduler num_train_timesteps %d", c.Scheduler.Params.NumTrainTimesteps)
	}
	return nil
}

func (c Hunyuan3DConfig) Summary() Hunyuan3DSummary {
	condType := c.Conditioner.Params.MainImageEncoder.Type
	if condType == "" {
		condType = c.Conditioner.Params.AdditionalImageEncoder.Type
	}
	return Hunyuan3DSummary{
		DenoiserTarget:    c.Model.Target,
		VAETarget:         c.VAE.Target,
		ConditionerTarget: c.Conditioner.Target,
		SchedulerTarget:   c.Scheduler.Target,
		InChannels:        c.Model.Params.InChannels,
		ContextInDim:      c.Model.Params.ContextInDim,
		HiddenSize:        c.Model.Params.HiddenSize,
		NumHeads:          c.Model.Params.NumHeads,
		Depth:             c.Model.Params.Depth,
		DepthSingleBlocks: c.Model.Params.DepthSingleBlocks,
		VAELatents:        c.VAE.Params.NumLatents,
		VAEEmbedDim:       c.VAE.Params.EmbedDim,
		VAEWidth:          c.VAE.Params.Width,
		VAEHeads:          c.VAE.Params.Heads,
		ConditionerType:   condType,
		SchedulerSteps:    c.Scheduler.Params.NumTrainTimesteps,
	}
}

type Hunyuan3DTensorGroup string

const (
	Hunyuan3DTensorModel       Hunyuan3DTensorGroup = "model"
	Hunyuan3DTensorVAE         Hunyuan3DTensorGroup = "vae"
	Hunyuan3DTensorConditioner Hunyuan3DTensorGroup = "conditioner"
	Hunyuan3DTensorOther       Hunyuan3DTensorGroup = "other"
)

type Hunyuan3DTensorInventory struct {
	Total       int
	Model       int
	VAE         int
	Conditioner int
	Other       int
	Examples    map[Hunyuan3DTensorGroup][]string
}

func ClassifyHunyuan3DTensorName(name string) Hunyuan3DTensorGroup {
	prefix := name
	if i := strings.IndexByte(prefix, '.'); i >= 0 {
		prefix = prefix[:i]
	}
	switch prefix {
	case "model":
		return Hunyuan3DTensorModel
	case "vae":
		return Hunyuan3DTensorVAE
	case "conditioner":
		return Hunyuan3DTensorConditioner
	default:
		return Hunyuan3DTensorOther
	}
}

func SummarizeHunyuan3DTensors(names []string) Hunyuan3DTensorInventory {
	inv := Hunyuan3DTensorInventory{Total: len(names), Examples: map[Hunyuan3DTensorGroup][]string{}}
	for _, name := range names {
		group := ClassifyHunyuan3DTensorName(name)
		switch group {
		case Hunyuan3DTensorModel:
			inv.Model++
		case Hunyuan3DTensorVAE:
			inv.VAE++
		case Hunyuan3DTensorConditioner:
			inv.Conditioner++
		default:
			inv.Other++
		}
		if len(inv.Examples[group]) < 5 {
			inv.Examples[group] = append(inv.Examples[group], name)
		}
	}
	return inv
}
