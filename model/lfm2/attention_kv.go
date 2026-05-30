package lfm2

import "fmt"

// AttentionKVLayout captures the full-attention cache contract. LFM2 only
// allocates KV cache for full_attention layers; convolution layers use the
// separate ConvStateLayout.
type AttentionKVLayout struct {
	Layers         int   `json:"layers"`
	KVHeads        int   `json:"kv_heads"`
	HeadDim        int   `json:"head_dim"`
	FloatsPerToken int   `json:"floats_per_token"`
	LayerIndices   []int `json:"layer_indices"`
}

func NewAttentionKVLayout(cfg Config, schedule LayerSchedule) (AttentionKVLayout, error) {
	if err := cfg.Validate(); err != nil {
		return AttentionKVLayout{}, err
	}
	if len(schedule.Steps) == 0 {
		var err error
		schedule, err = NewLayerSchedule(cfg)
		if err != nil {
			return AttentionKVLayout{}, err
		}
	}
	layout := AttentionKVLayout{
		Layers:       len(schedule.FullAttentionIndices),
		KVHeads:      cfg.NumKeyValueHeads,
		HeadDim:      cfg.HeadDim,
		LayerIndices: append([]int(nil), schedule.FullAttentionIndices...),
	}
	layout.FloatsPerToken = 2 * layout.Layers * layout.KVHeads * layout.HeadDim
	return layout, layout.Validate()
}

func (l AttentionKVLayout) Validate() error {
	if l.KVHeads <= 0 || l.HeadDim <= 0 || l.Layers < 0 {
		return fmt.Errorf("invalid LFM2 attention KV layout dims: %+v", l)
	}
	if l.Layers == 0 {
		if len(l.LayerIndices) != 0 || l.FloatsPerToken != 0 {
			return fmt.Errorf("invalid empty LFM2 attention KV layout: %+v", l)
		}
		return nil
	}
	if len(l.LayerIndices) != l.Layers {
		return fmt.Errorf("invalid LFM2 attention layer index count=%d want=%d", len(l.LayerIndices), l.Layers)
	}
	wantFloats := 2 * l.Layers * l.KVHeads * l.HeadDim
	if l.FloatsPerToken != wantFloats {
		return fmt.Errorf("invalid LFM2 attention KV floats/token=%d want=%d", l.FloatsPerToken, wantFloats)
	}
	prev := -1
	for _, idx := range l.LayerIndices {
		if idx <= prev {
			return fmt.Errorf("invalid LFM2 attention layer order: %v", l.LayerIndices)
		}
		prev = idx
	}
	return nil
}

func (l AttentionKVLayout) Bytes(maxSeq int, bytesPerFloat int) (int64, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if maxSeq < 0 || bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid attention KV sizing arguments: max_seq=%d bytes_per_float=%d", maxSeq, bytesPerFloat)
	}
	return int64(maxSeq) * int64(l.FloatsPerToken) * int64(bytesPerFloat), nil
}
