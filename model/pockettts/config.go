// Package pockettts implements Kyutai Pocket TTS in native Go/SIMD.
package pockettts

import (
	"fmt"
	"math"

	"gopkg.in/yaml.v3"
)

const (
	UpstreamCommit           = "0acce6b2f390150267557770d2098c5caa9a18ac"
	EnglishWeightsRevision   = "e7205b6ee50e654a5ea19f0e9df2b0813b05e921"
	EnglishTokenizerRevision = "00eac05ed3d16bdc3f6b5d598874019c34a89214"
	SampleRate               = 24000
	FrameRateNumerator       = 25
	FrameRateDenominator     = 2
	SamplesPerFrame          = 1920
)

type Config struct {
	WeightsPath                    string       `yaml:"weights_path"`
	WeightsPathWithoutVoiceCloning string       `yaml:"weights_path_without_voice_cloning"`
	DefaultTemperature             float64      `yaml:"default_temperature"`
	FlowLM                         FlowLMConfig `yaml:"flow_lm"`
	Mimi                           MimiConfig   `yaml:"mimi"`
}

type FlowLMConfig struct {
	InsertBOSBeforeVoice bool              `yaml:"insert_bos_before_voice"`
	DType                string            `yaml:"dtype"`
	Flow                 FlowConfig        `yaml:"flow"`
	Transformer          TransformerConfig `yaml:"transformer"`
	LookupTable          LookupTableConfig `yaml:"lookup_table"`
}

type FlowConfig struct {
	Type  string `yaml:"type"`
	Depth int    `yaml:"depth"`
	Dim   int    `yaml:"dim"`
}

type TransformerConfig struct {
	DModel           int     `yaml:"d_model"`
	HiddenScale      int     `yaml:"hidden_scale"`
	MaxPeriod        float64 `yaml:"max_period"`
	NumHeads         int     `yaml:"num_heads"`
	NumLayers        int     `yaml:"num_layers"`
	LayerScale       float64 `yaml:"layer_scale"`
	Context          int     `yaml:"context"`
	DimFeedforward   int     `yaml:"dim_feedforward"`
	InputDimension   int     `yaml:"input_dimension"`
	OutputDimensions []int   `yaml:"output_dimensions"`
}

type LookupTableConfig struct {
	Dim           int    `yaml:"dim"`
	NBins         int    `yaml:"n_bins"`
	Tokenizer     string `yaml:"tokenizer"`
	TokenizerPath string `yaml:"tokenizer_path"`
}

type MimiConfig struct {
	DType       string            `yaml:"dtype"`
	SampleRate  int               `yaml:"sample_rate"`
	InnerDim    int               `yaml:"inner_dim"`
	OuterDim    int               `yaml:"outer_dim"`
	Channels    int               `yaml:"channels"`
	FrameRate   float64           `yaml:"frame_rate"`
	SEANet      SEANetConfig      `yaml:"seanet"`
	Transformer TransformerConfig `yaml:"transformer"`
	Quantizer   QuantizerConfig   `yaml:"quantizer"`
}

type SEANetConfig struct {
	Dimension          int    `yaml:"dimension"`
	Channels           int    `yaml:"channels"`
	NFilters           int    `yaml:"n_filters"`
	NResidualLayers    int    `yaml:"n_residual_layers"`
	Ratios             []int  `yaml:"ratios"`
	KernelSize         int    `yaml:"kernel_size"`
	ResidualKernelSize int    `yaml:"residual_kernel_size"`
	LastKernelSize     int    `yaml:"last_kernel_size"`
	DilationBase       int    `yaml:"dilation_base"`
	PadMode            string `yaml:"pad_mode"`
	Compress           int    `yaml:"compress"`
}

type QuantizerConfig struct {
	Dimension       int `yaml:"dimension"`
	OutputDimension int `yaml:"output_dimension"`
}

func ParseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse Pocket TTS config: %w", err)
	}
	if cfg.FlowLM.Flow.Type == "" {
		cfg.FlowLM.Flow.Type = "lsd"
	}
	return cfg, cfg.Validate()
}

func (c Config) Validate() error {
	f, m := c.FlowLM, c.Mimi
	if c.WeightsPath == "" && c.WeightsPathWithoutVoiceCloning == "" {
		return fmt.Errorf("Pocket TTS config has no weights path")
	}
	if math.IsNaN(c.DefaultTemperature) || math.IsInf(c.DefaultTemperature, 0) || c.DefaultTemperature <= 0 {
		return fmt.Errorf("invalid Pocket TTS temperature %g", c.DefaultTemperature)
	}
	if f.DType != "float32" || !f.InsertBOSBeforeVoice || f.Flow.Depth <= 0 || f.Flow.Dim <= 0 || (f.Flow.Type != "lsd" && f.Flow.Type != "flow_matching") {
		return fmt.Errorf("invalid Pocket TTS FlowLM config: %+v", f)
	}
	if err := validateTransformer("FlowLM", f.Transformer, false); err != nil {
		return err
	}
	if f.LookupTable.Dim != f.Transformer.DModel || f.LookupTable.NBins <= 0 || f.LookupTable.TokenizerPath == "" {
		return fmt.Errorf("invalid Pocket TTS lookup table: %+v", f.LookupTable)
	}
	if m.DType != "float32" || m.SampleRate != SampleRate || m.Channels != 1 || m.FrameRate != float64(FrameRateNumerator)/FrameRateDenominator || m.InnerDim <= 0 || m.OuterDim <= 0 {
		return fmt.Errorf("invalid Pocket TTS Mimi geometry: %+v", m)
	}
	if err := validateTransformer("Mimi", m.Transformer, true); err != nil {
		return err
	}
	if m.SEANet.Dimension <= 0 || m.SEANet.Channels != 1 || m.SEANet.NFilters <= 0 || m.SEANet.NResidualLayers <= 0 || len(m.SEANet.Ratios) == 0 || m.SEANet.Compress <= 0 {
		return fmt.Errorf("invalid Pocket TTS SEANet: %+v", m.SEANet)
	}
	hop := 1
	for _, ratio := range m.SEANet.Ratios {
		if ratio <= 0 || hop > math.MaxInt/ratio {
			return fmt.Errorf("invalid Pocket TTS SEANet ratios %v", m.SEANet.Ratios)
		}
		hop *= ratio
	}
	encoderRate := m.SampleRate / hop
	if m.SampleRate%hop != 0 || encoderRate*FrameRateDenominator%FrameRateNumerator != 0 || encoderRate*FrameRateDenominator/FrameRateNumerator != 16 {
		return fmt.Errorf("invalid Pocket TTS Mimi rates sample=%d hop=%d frame=%g", m.SampleRate, hop, m.FrameRate)
	}
	if m.Quantizer.Dimension != m.InnerDim || m.Quantizer.OutputDimension != m.OuterDim {
		return fmt.Errorf("invalid Pocket TTS quantizer: %+v", m.Quantizer)
	}
	return nil
}

func validateTransformer(name string, t TransformerConfig, projected bool) error {
	if t.DModel <= 0 || t.NumHeads <= 0 || t.NumLayers <= 0 || t.DModel%t.NumHeads != 0 {
		return fmt.Errorf("invalid Pocket TTS %s transformer: %+v", name, t)
	}
	if projected {
		if t.DimFeedforward <= 0 || t.Context <= 0 || t.InputDimension <= 0 || len(t.OutputDimensions) == 0 {
			return fmt.Errorf("invalid Pocket TTS %s projected transformer: %+v", name, t)
		}
	} else if t.HiddenScale <= 0 || t.MaxPeriod <= 0 {
		return fmt.Errorf("invalid Pocket TTS %s transformer scale/period: %+v", name, t)
	}
	return nil
}

func (c Config) SamplesForFrames(frames int) (int, error) {
	if err := c.Validate(); err != nil {
		return 0, err
	}
	if frames <= 0 || frames > math.MaxInt/SamplesPerFrame {
		return 0, fmt.Errorf("invalid Pocket TTS frame count %d", frames)
	}
	return frames * SamplesPerFrame, nil
}
