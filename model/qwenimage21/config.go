// Package qwenimage21 provides native Qwen Image 2.1 model contracts and
// reusable CPU/SIMD primitives. Full generation is deliberately gated on all
// required components and exact checkpoint validation.
package qwenimage21

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	StableDiffusionCPPSourcePin  = "c678dfe704a2230342376b46add9c8ca736a653d"
	StableDiffusionCPPFeaturePin = "137f7409bbfb98c70a350a57d6a135487080db96"
	OfficialModelPin             = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
	ComfyModelPin                = "ace0edeb3791a594ddfa36ed5f41a178a394e921"
	GGUFModelPin                 = "cc11433936a06e9765f7c0c0b1f0436cfd2b9856"
)

type ModelIndex struct {
	ClassName   string   `json:"_class_name"`
	Processor   []string `json:"processor"`
	Scheduler   []string `json:"scheduler"`
	TextEncoder []string `json:"text_encoder"`
	Transformer []string `json:"transformer"`
	VAE         []string `json:"vae"`
}
type TransformerConfig struct {
	ClassName         string  `json:"_class_name"`
	AttentionHeadDim  int     `json:"attention_head_dim"`
	AxesDimsRoPE      []int   `json:"axes_dims_rope"`
	ContextInDim      int     `json:"context_in_dim"`
	InChannels        int     `json:"in_channels"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumLayers         int     `json:"num_layers"`
	OutChannels       int     `json:"out_channels"`
	PatchSize         int     `json:"patch_size"`
	MLPRatio          int     `json:"mlp_ratio"`
	Eps               float64 `json:"eps"`
	CausalCondition   bool    `json:"causal_condition"`
}
type SchedulerConfig struct {
	ClassName          string  `json:"_class_name"`
	BaseImageSeqLen    int     `json:"base_image_seq_len"`
	BaseShift          float64 `json:"base_shift"`
	MaxImageSeqLen     int     `json:"max_image_seq_len"`
	MaxShift           float64 `json:"max_shift"`
	NumTrainTimesteps  int     `json:"num_train_timesteps"`
	Shift              float64 `json:"shift"`
	ShiftTerminal      float64 `json:"shift_terminal"`
	TimeShiftType      string  `json:"time_shift_type"`
	UseDynamicShifting bool    `json:"use_dynamic_shifting"`
	InvertSigmas       bool    `json:"invert_sigmas"`
	StochasticSampling bool    `json:"stochastic_sampling"`
}
type VAEConfig struct {
	ClassName           string    `json:"_class_name"`
	InChannels          int       `json:"in_channels"`
	OutChannels         int       `json:"out_channels"`
	BaseDim             int       `json:"base_dim"`
	DecoderBaseDim      int       `json:"decoder_base_dim"`
	DimMult             []int     `json:"dim_mult"`
	NumResBlocks        int       `json:"num_res_blocks"`
	ZDim                int       `json:"z_dim"`
	ScaleFactorSpatial  int       `json:"scale_factor_spatial"`
	ScaleFactorTemporal int       `json:"scale_factor_temporal"`
	LatentsMean         []float64 `json:"latents_mean"`
	LatentsStd          []float64 `json:"latents_std"`
	TemporalDownsample  []bool    `json:"temperal_downsample"`
}
type Config struct {
	Root        string
	ModelIndex  ModelIndex
	Transformer TransformerConfig
	Scheduler   SchedulerConfig
	VAE         VAEConfig
}

func ReadConfig(root string) (Config, error) {
	var c Config
	c.Root = root
	for _, item := range []struct {
		path string
		dst  any
	}{{"model_index.json", &c.ModelIndex}, {"transformer/config.json", &c.Transformer}, {"scheduler/scheduler_config.json", &c.Scheduler}, {"vae/config.json", &c.VAE}} {
		b, e := os.ReadFile(filepath.Join(root, item.path))
		if e != nil {
			return c, e
		}
		if e = json.Unmarshal(b, item.dst); e != nil {
			return c, fmt.Errorf("%s: %w", item.path, e)
		}
	}
	return c, ValidateConfig(c)
}
func ValidateConfig(c Config) error {
	m := c.ModelIndex
	t := c.Transformer
	s := c.Scheduler
	v := c.VAE
	if m.ClassName != "QwenImage21Pipeline" || last(m.Processor) != "Qwen3VLProcessor" || last(m.TextEncoder) != "Qwen3VLForConditionalGeneration" || last(m.Transformer) != "QwenImage21Transformer2DModel" || last(m.VAE) != "AutoencoderKLQwenImage21" || last(m.Scheduler) != "FlowMatchEulerDiscreteScheduler" {
		return fmt.Errorf("qwen-image-2.1: unsupported pipeline components")
	}
	if t.ClassName != "QwenImage21Transformer2DModel" || t.InChannels != 64 || t.OutChannels != 64 || t.NumLayers != 32 || t.NumAttentionHeads != 32 || t.AttentionHeadDim != 128 || t.ContextInDim != 4096 || t.PatchSize != 1 || t.MLPRatio != 3 || !t.CausalCondition || len(t.AxesDimsRoPE) != 3 || t.AxesDimsRoPE[0]+t.AxesDimsRoPE[1]+t.AxesDimsRoPE[2] != 128 {
		return fmt.Errorf("qwen-image-2.1: unsupported transformer config")
	}
	if s.ClassName != "FlowMatchEulerDiscreteScheduler" || s.NumTrainTimesteps != 1000 || !s.UseDynamicShifting || s.TimeShiftType != "exponential" || s.BaseImageSeqLen <= 0 || s.MaxImageSeqLen <= s.BaseImageSeqLen || s.ShiftTerminal <= 0 || s.ShiftTerminal >= 1 || s.InvertSigmas || s.StochasticSampling {
		return fmt.Errorf("qwen-image-2.1: unsupported scheduler config")
	}
	if v.ClassName != "AutoencoderKLQwenImage21" || v.InChannels != 4 || v.OutChannels != 4 || v.ZDim != 64 || v.ScaleFactorSpatial != 16 || len(v.LatentsMean) != 64 || len(v.LatentsStd) != 64 || len(v.DimMult) != 5 {
		return fmt.Errorf("qwen-image-2.1: unsupported VAE config")
	}
	for _, x := range v.LatentsStd {
		if x <= 0 {
			return fmt.Errorf("qwen-image-2.1: invalid VAE latent std")
		}
	}
	return nil
}
func last(v []string) string {
	if len(v) == 0 {
		return ""
	}
	return v[len(v)-1]
}
func (c Config) HiddenSize() int {
	return c.Transformer.NumAttentionHeads * c.Transformer.AttentionHeadDim
}
func (c Config) IntermediateSize() int { return c.HiddenSize() * c.Transformer.MLPRatio }
func (c Config) LatentGrid(height, width int) (int, int, error) {
	if e := ValidateConfig(c); e != nil {
		return 0, 0, e
	}
	if height <= 0 || width <= 0 || height%c.VAE.ScaleFactorSpatial != 0 || width%c.VAE.ScaleFactorSpatial != 0 {
		return 0, 0, fmt.Errorf("qwen-image-2.1: dimensions must be positive multiples of %d", c.VAE.ScaleFactorSpatial)
	}
	return height / c.VAE.ScaleFactorSpatial, width / c.VAE.ScaleFactorSpatial, nil
}
