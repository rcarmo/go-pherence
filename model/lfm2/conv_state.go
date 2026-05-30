package lfm2

import "fmt"

// ConvStateLayout captures the per-layer convolution cache contract. LFM2.5
// advertises a small conv_L_cache; runtime code should keep exactly this many
// hidden vectors for every convolution layer.
type ConvStateLayout struct {
	Layers         int   `json:"layers"`
	HiddenSize     int   `json:"hidden_size"`
	LCache         int   `json:"l_cache"`
	FloatsPerLayer int   `json:"floats_per_layer"`
	TotalFloats    int   `json:"total_floats"`
	LayerIndices   []int `json:"layer_indices"`
}

func NewConvStateLayout(cfg Config, schedule LayerSchedule) (ConvStateLayout, error) {
	if err := cfg.Validate(); err != nil {
		return ConvStateLayout{}, err
	}
	if len(schedule.Steps) == 0 {
		var err error
		schedule, err = NewLayerSchedule(cfg)
		if err != nil {
			return ConvStateLayout{}, err
		}
	}
	layout := ConvStateLayout{
		Layers:         len(schedule.ConvIndices),
		HiddenSize:     cfg.HiddenSize,
		LCache:         cfg.ConvLCache,
		FloatsPerLayer: cfg.ConvLCache * cfg.HiddenSize,
		LayerIndices:   append([]int(nil), schedule.ConvIndices...),
	}
	layout.TotalFloats = layout.Layers * layout.FloatsPerLayer
	return layout, layout.Validate()
}

func (l ConvStateLayout) Validate() error {
	if l.Layers <= 0 || l.HiddenSize <= 0 || l.LCache <= 0 {
		return fmt.Errorf("invalid LFM2 conv state layout dims: %+v", l)
	}
	if len(l.LayerIndices) != l.Layers {
		return fmt.Errorf("invalid LFM2 conv state layer index count=%d want=%d", len(l.LayerIndices), l.Layers)
	}
	if l.FloatsPerLayer != l.LCache*l.HiddenSize {
		return fmt.Errorf("invalid LFM2 conv state floats/layer=%d want=%d", l.FloatsPerLayer, l.LCache*l.HiddenSize)
	}
	if l.TotalFloats != l.Layers*l.FloatsPerLayer {
		return fmt.Errorf("invalid LFM2 conv state total_floats=%d want=%d", l.TotalFloats, l.Layers*l.FloatsPerLayer)
	}
	prev := -1
	for _, idx := range l.LayerIndices {
		if idx <= prev {
			return fmt.Errorf("invalid LFM2 conv state layer order: %v", l.LayerIndices)
		}
		prev = idx
	}
	return nil
}

func (l ConvStateLayout) Bytes(bytesPerFloat int) (int64, error) {
	if bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid conv state bytes/float=%d", bytesPerFloat)
	}
	return int64(l.TotalFloats) * int64(bytesPerFloat), nil
}
