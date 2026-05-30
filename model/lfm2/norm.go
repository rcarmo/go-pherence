package lfm2

import "fmt"

// NormLayout captures LFM2 normalization dimensions and epsilon. Runtime code
// should use this contract for token embedding norm, attention norm, and FFN
// norm buffers rather than inferring sizes ad hoc.
type NormLayout struct {
	HiddenSize       int     `json:"hidden_size"`
	Layers           int     `json:"layers"`
	Epsilon          float64 `json:"epsilon"`
	NormsPerLayer    int     `json:"norms_per_layer"`
	TotalNormVectors int     `json:"total_norm_vectors"`
	FloatsPerVector  int     `json:"floats_per_vector"`
}

func NewNormLayout(cfg Config) (NormLayout, error) {
	if err := cfg.Validate(); err != nil {
		return NormLayout{}, err
	}
	eps := cfg.NormEps
	if eps == 0 {
		eps = 1e-5
	}
	layout := NormLayout{HiddenSize: cfg.HiddenSize, Layers: cfg.NumHiddenLayers, Epsilon: eps, NormsPerLayer: 2, FloatsPerVector: cfg.HiddenSize}
	layout.TotalNormVectors = layout.Layers * layout.NormsPerLayer
	return layout, layout.Validate()
}

func (l NormLayout) Validate() error {
	if l.HiddenSize <= 0 || l.Layers <= 0 || l.Epsilon <= 0 || l.NormsPerLayer <= 0 || l.FloatsPerVector != l.HiddenSize {
		return fmt.Errorf("invalid LFM2 norm layout: %+v", l)
	}
	if l.TotalNormVectors != l.Layers*l.NormsPerLayer {
		return fmt.Errorf("invalid LFM2 norm vector count=%d want=%d", l.TotalNormVectors, l.Layers*l.NormsPerLayer)
	}
	return nil
}

func (l NormLayout) ScratchFloats(tokens int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if tokens < 0 {
		return 0, fmt.Errorf("invalid LFM2 norm token count=%d", tokens)
	}
	return tokens * l.HiddenSize, nil
}
