package lfm2

import "fmt"

// RoPELayout captures the positional encoding contract for LFM2 full-attention
// layers. Convolution layers do not allocate attention KV, but the same context
// limit bounds all token positions.
type RoPELayout struct {
	Theta                 float64 `json:"theta"`
	Type                  string  `json:"type"`
	HeadDim               int     `json:"head_dim"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	FullAttentionLayers   int     `json:"full_attention_layers"`
}

func NewRoPELayout(cfg Config, schedule LayerSchedule) (RoPELayout, error) {
	if err := cfg.Validate(); err != nil {
		return RoPELayout{}, err
	}
	if len(schedule.Steps) == 0 {
		var err error
		schedule, err = NewLayerSchedule(cfg)
		if err != nil {
			return RoPELayout{}, err
		}
	}
	maxPos := cfg.MaxPositionEmbeddings
	if maxPos == 0 {
		maxPos = 128000
	}
	layout := RoPELayout{Theta: cfg.RoPE.Theta, Type: cfg.RoPE.Type, HeadDim: cfg.HeadDim, MaxPositionEmbeddings: maxPos, FullAttentionLayers: len(schedule.FullAttentionIndices)}
	return layout, layout.Validate()
}

func (l RoPELayout) Validate() error {
	if l.Theta < 0 || l.HeadDim <= 0 || l.MaxPositionEmbeddings <= 0 || l.FullAttentionLayers < 0 {
		return fmt.Errorf("invalid LFM2 RoPE layout: %+v", l)
	}
	return nil
}

func (l RoPELayout) ValidatePosition(pos int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if pos < 0 || pos >= l.MaxPositionEmbeddings {
		return fmt.Errorf("invalid LFM2 RoPE position=%d max=%d", pos, l.MaxPositionEmbeddings)
	}
	return nil
}
