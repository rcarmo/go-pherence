package lfm2

import "fmt"

// AttentionProjectionLayout captures LFM2 full-attention projection matrix
// dimensions. It makes the Q/O hidden-width projections and GQA K/V widths
// explicit before attention runtime code starts binding tensor names to kernels.
type AttentionProjectionLayout struct {
	HiddenSize           int `json:"hidden_size"`
	Heads                int `json:"heads"`
	KVHeads              int `json:"kv_heads"`
	HeadDim              int `json:"head_dim"`
	QueriesPerKV         int `json:"queries_per_kv"`
	QLayerFloats         int `json:"q_layer_floats"`
	KLayerFloats         int `json:"k_layer_floats"`
	VLayerFloats         int `json:"v_layer_floats"`
	OLayerFloats         int `json:"o_layer_floats"`
	TotalFloatsPerLayer  int `json:"total_floats_per_layer"`
	FullAttentionLayers  int `json:"full_attention_layers"`
	TotalAttentionFloats int `json:"total_attention_floats"`
}

func NewAttentionProjectionLayout(cfg Config, schedule LayerSchedule) (AttentionProjectionLayout, error) {
	if err := cfg.Validate(); err != nil {
		return AttentionProjectionLayout{}, err
	}
	if len(schedule.Steps) == 0 {
		var err error
		schedule, err = NewLayerSchedule(cfg)
		if err != nil {
			return AttentionProjectionLayout{}, err
		}
	}
	if err := schedule.Validate(cfg.NumHiddenLayers); err != nil {
		return AttentionProjectionLayout{}, err
	}
	kvWidth := cfg.NumKeyValueHeads * cfg.HeadDim
	layout := AttentionProjectionLayout{
		HiddenSize:          cfg.HiddenSize,
		Heads:               cfg.NumAttentionHeads,
		KVHeads:             cfg.NumKeyValueHeads,
		HeadDim:             cfg.HeadDim,
		QueriesPerKV:        cfg.NumAttentionHeads / cfg.NumKeyValueHeads,
		QLayerFloats:        cfg.HiddenSize * cfg.HiddenSize,
		KLayerFloats:        cfg.HiddenSize * kvWidth,
		VLayerFloats:        cfg.HiddenSize * kvWidth,
		OLayerFloats:        cfg.HiddenSize * cfg.HiddenSize,
		FullAttentionLayers: len(schedule.FullAttentionIndices),
	}
	layout.TotalFloatsPerLayer = layout.QLayerFloats + layout.KLayerFloats + layout.VLayerFloats + layout.OLayerFloats
	layout.TotalAttentionFloats = layout.TotalFloatsPerLayer * layout.FullAttentionLayers
	return layout, layout.Validate()
}

func (l AttentionProjectionLayout) Validate() error {
	if l.HiddenSize <= 0 || l.Heads <= 0 || l.KVHeads <= 0 || l.HeadDim <= 0 || l.FullAttentionLayers < 0 {
		return fmt.Errorf("invalid LFM2 attention projection dims: %+v", l)
	}
	if l.HiddenSize != l.Heads*l.HeadDim {
		return fmt.Errorf("invalid LFM2 attention projection head dims: hidden=%d heads=%d head_dim=%d", l.HiddenSize, l.Heads, l.HeadDim)
	}
	if l.Heads%l.KVHeads != 0 {
		return fmt.Errorf("invalid LFM2 attention projection GQA dims: heads=%d kv_heads=%d", l.Heads, l.KVHeads)
	}
	if l.QueriesPerKV != l.Heads/l.KVHeads {
		return fmt.Errorf("invalid LFM2 attention queries/kv=%d want=%d", l.QueriesPerKV, l.Heads/l.KVHeads)
	}
	kvWidth := l.KVHeads * l.HeadDim
	wantQO := l.HiddenSize * l.HiddenSize
	wantKV := l.HiddenSize * kvWidth
	if l.QLayerFloats != wantQO || l.OLayerFloats != wantQO || l.KLayerFloats != wantKV || l.VLayerFloats != wantKV {
		return fmt.Errorf("invalid LFM2 attention projection floats: %+v", l)
	}
	wantLayer := l.QLayerFloats + l.KLayerFloats + l.VLayerFloats + l.OLayerFloats
	if l.TotalFloatsPerLayer != wantLayer {
		return fmt.Errorf("invalid LFM2 attention projection floats/layer=%d want=%d", l.TotalFloatsPerLayer, wantLayer)
	}
	if l.TotalAttentionFloats != l.TotalFloatsPerLayer*l.FullAttentionLayers {
		return fmt.Errorf("invalid LFM2 attention projection total=%d want=%d", l.TotalAttentionFloats, l.TotalFloatsPerLayer*l.FullAttentionLayers)
	}
	return nil
}
