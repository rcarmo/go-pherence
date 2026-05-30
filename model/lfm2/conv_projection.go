package lfm2

import "fmt"

// ConvProjectionLayout captures the depthwise short-convolution parameters used
// by LFM2 conv layers. The recurrent cache layout is covered separately by
// ConvStateLayout; this contract sizes the per-layer convolution kernel/bias
// tensors that runtime code will bind before implementing the conv block.
type ConvProjectionLayout struct {
	HiddenSize      int  `json:"hidden_size"`
	ConvLCache      int  `json:"conv_l_cache"`
	ConvLayers      int  `json:"conv_layers"`
	HasBias         bool `json:"has_bias"`
	KernelFloats    int  `json:"kernel_floats"`
	BiasFloats      int  `json:"bias_floats"`
	FloatsPerLayer  int  `json:"floats_per_layer"`
	TotalConvFloats int  `json:"total_conv_floats"`
}

func NewConvProjectionLayout(cfg Config, schedule LayerSchedule) (ConvProjectionLayout, error) {
	if err := cfg.Validate(); err != nil {
		return ConvProjectionLayout{}, err
	}
	if len(schedule.Steps) == 0 {
		var err error
		schedule, err = NewLayerSchedule(cfg)
		if err != nil {
			return ConvProjectionLayout{}, err
		}
	}
	if err := schedule.Validate(cfg.NumHiddenLayers); err != nil {
		return ConvProjectionLayout{}, err
	}
	layout := ConvProjectionLayout{
		HiddenSize:   cfg.HiddenSize,
		ConvLCache:   cfg.ConvLCache,
		ConvLayers:   len(schedule.ConvIndices),
		HasBias:      cfg.ConvBias,
		KernelFloats: cfg.HiddenSize * cfg.ConvLCache,
	}
	if cfg.ConvBias {
		layout.BiasFloats = cfg.HiddenSize
	}
	layout.FloatsPerLayer = layout.KernelFloats + layout.BiasFloats
	layout.TotalConvFloats = layout.FloatsPerLayer * layout.ConvLayers
	return layout, layout.Validate()
}

func (l ConvProjectionLayout) Validate() error {
	if l.HiddenSize <= 0 || l.ConvLCache <= 0 || l.ConvLayers < 0 {
		return fmt.Errorf("invalid LFM2 conv projection dims: %+v", l)
	}
	wantKernel := l.HiddenSize * l.ConvLCache
	if l.KernelFloats != wantKernel {
		return fmt.Errorf("invalid LFM2 conv kernel floats=%d want=%d", l.KernelFloats, wantKernel)
	}
	wantBias := 0
	if l.HasBias {
		wantBias = l.HiddenSize
	}
	if l.BiasFloats != wantBias {
		return fmt.Errorf("invalid LFM2 conv bias floats=%d want=%d", l.BiasFloats, wantBias)
	}
	wantLayer := l.KernelFloats + l.BiasFloats
	if l.FloatsPerLayer != wantLayer {
		return fmt.Errorf("invalid LFM2 conv floats/layer=%d want=%d", l.FloatsPerLayer, wantLayer)
	}
	if l.TotalConvFloats != l.FloatsPerLayer*l.ConvLayers {
		return fmt.Errorf("invalid LFM2 conv total floats=%d want=%d", l.TotalConvFloats, l.FloatsPerLayer*l.ConvLayers)
	}
	return nil
}
