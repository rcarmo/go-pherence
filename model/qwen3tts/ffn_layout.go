package qwen3tts

import "fmt"

// FFNLayout captures the gated feed-forward projection sizing for a Qwen3-TTS
// transformer stack. Qwen-style MLP blocks use gate/up/down projections, so the
// runtime can validate three matrices per layer before binding weights.
type FFNLayout struct {
	Name                 string `json:"name"`
	HiddenSize           int    `json:"hidden_size"`
	IntermediateSize     int    `json:"intermediate_size"`
	Layers               int    `json:"layers"`
	GateProjectionFloats int    `json:"gate_projection_floats"`
	UpProjectionFloats   int    `json:"up_projection_floats"`
	DownProjectionFloats int    `json:"down_projection_floats"`
	FloatsPerLayer       int    `json:"floats_per_layer"`
	TotalFloats          int    `json:"total_floats"`
}

func NewTalkerFFNLayout(cfg ParsedConfig) (FFNLayout, error) {
	if err := cfg.Validate(); err != nil {
		return FFNLayout{}, err
	}
	layout := newFFNLayout("talker", cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize, cfg.TalkerNumHiddenLayers)
	return layout, layout.Validate()
}

func NewCodePredictorFFNLayout(cfg ParsedConfig) (FFNLayout, error) {
	if err := cfg.Validate(); err != nil {
		return FFNLayout{}, err
	}
	layout := newFFNLayout("code_predictor", cfg.CPHiddenSize, cfg.CPIntermediateSize, cfg.CPNumHiddenLayers)
	return layout, layout.Validate()
}

func newFFNLayout(name string, hidden, intermediate, layers int) FFNLayout {
	proj := sizeProduct(hidden, intermediate)
	layout := FFNLayout{
		Name:                 name,
		HiddenSize:           hidden,
		IntermediateSize:     intermediate,
		Layers:               layers,
		GateProjectionFloats: proj,
		UpProjectionFloats:   proj,
		DownProjectionFloats: proj,
	}
	layout.FloatsPerLayer = sizeSum(layout.GateProjectionFloats, layout.UpProjectionFloats, layout.DownProjectionFloats)
	layout.TotalFloats = sizeProduct(layout.FloatsPerLayer, layers)
	return layout
}

func (l FFNLayout) Validate() error {
	if l.Name == "" || l.HiddenSize <= 0 || l.IntermediateSize <= 0 || l.Layers <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS FFN layout dims: %+v", l)
	}
	wantProj := sizeProduct(l.HiddenSize, l.IntermediateSize)
	if wantProj < 0 || l.GateProjectionFloats != wantProj || l.UpProjectionFloats != wantProj || l.DownProjectionFloats != wantProj {
		return fmt.Errorf("invalid Qwen3-TTS %s FFN projection floats: %+v", l.Name, l)
	}
	wantLayer := sizeSum(l.GateProjectionFloats, l.UpProjectionFloats, l.DownProjectionFloats)
	if wantLayer < 0 || l.FloatsPerLayer != wantLayer {
		return fmt.Errorf("invalid Qwen3-TTS %s FFN floats/layer=%d want=%d", l.Name, l.FloatsPerLayer, wantLayer)
	}
	wantTotal := sizeProduct(l.FloatsPerLayer, l.Layers)
	if wantTotal < 0 || l.TotalFloats != wantTotal {
		return fmt.Errorf("invalid Qwen3-TTS %s FFN total floats=%d want=%d", l.Name, l.TotalFloats, wantTotal)
	}
	return nil
}
