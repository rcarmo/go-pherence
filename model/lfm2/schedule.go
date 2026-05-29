package lfm2

import "fmt"

type LayerKind string

const (
	LayerConv          LayerKind = "conv"
	LayerFullAttention LayerKind = "full_attention"
)

type LayerStep struct {
	Index int       `json:"index"`
	Kind  LayerKind `json:"kind"`
}

type LayerSchedule struct {
	Steps                []LayerStep `json:"steps"`
	ConvIndices          []int       `json:"conv_indices"`
	FullAttentionIndices []int       `json:"full_attention_indices"`
}

func NewLayerSchedule(cfg Config) (LayerSchedule, error) {
	if err := cfg.Validate(); err != nil {
		return LayerSchedule{}, err
	}
	s := LayerSchedule{Steps: make([]LayerStep, 0, len(cfg.LayerTypes))}
	for i, kind := range cfg.LayerTypes {
		step := LayerStep{Index: i, Kind: LayerKind(kind)}
		s.Steps = append(s.Steps, step)
		switch step.Kind {
		case LayerConv:
			s.ConvIndices = append(s.ConvIndices, i)
		case LayerFullAttention:
			s.FullAttentionIndices = append(s.FullAttentionIndices, i)
		default:
			return LayerSchedule{}, fmt.Errorf("invalid LFM2 layer kind at %d: %q", i, kind)
		}
	}
	if err := s.Validate(cfg.NumHiddenLayers); err != nil {
		return LayerSchedule{}, err
	}
	return s, nil
}

func (s LayerSchedule) Validate(numLayers int) error {
	if numLayers <= 0 || len(s.Steps) != numLayers {
		return fmt.Errorf("invalid LFM2 schedule length=%d want=%d", len(s.Steps), numLayers)
	}
	seen := make([]bool, numLayers)
	conv, attn := 0, 0
	for pos, step := range s.Steps {
		if step.Index != pos {
			return fmt.Errorf("invalid LFM2 schedule position=%d index=%d", pos, step.Index)
		}
		if step.Index < 0 || step.Index >= numLayers {
			return fmt.Errorf("invalid LFM2 schedule index=%d layers=%d", step.Index, numLayers)
		}
		if seen[step.Index] {
			return fmt.Errorf("duplicate LFM2 schedule index=%d", step.Index)
		}
		seen[step.Index] = true
		switch step.Kind {
		case LayerConv:
			conv++
		case LayerFullAttention:
			attn++
		default:
			return fmt.Errorf("invalid LFM2 schedule kind at %d: %q", step.Index, step.Kind)
		}
	}
	if conv != len(s.ConvIndices) || attn != len(s.FullAttentionIndices) {
		return fmt.Errorf("invalid LFM2 schedule index counts: conv=%d/%d attn=%d/%d", conv, len(s.ConvIndices), attn, len(s.FullAttentionIndices))
	}
	return nil
}

func (s LayerSchedule) IsFullAttentionLayer(layer int) bool {
	for _, idx := range s.FullAttentionIndices {
		if idx == layer {
			return true
		}
	}
	return false
}

func (s LayerSchedule) IsConvLayer(layer int) bool {
	for _, idx := range s.ConvIndices {
		if idx == layer {
			return true
		}
	}
	return false
}
