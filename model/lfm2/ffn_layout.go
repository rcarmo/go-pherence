package lfm2

import "fmt"

// FFNLayout captures dense and routed-MLP dimensions without executing them.
// Dense warmup layers use IntermediateSize; routed MoE layers use
// MoEIntermediateSize for each expert.
type FFNLayout struct {
	HiddenSize            int `json:"hidden_size"`
	DenseIntermediate     int `json:"dense_intermediate"`
	MoEIntermediate       int `json:"moe_intermediate"`
	DenseLayers           int `json:"dense_layers"`
	MoELayers             int `json:"moe_layers"`
	Experts               int `json:"experts"`
	ExpertsPerToken       int `json:"experts_per_token"`
	DenseParamsPerLayer   int `json:"dense_params_per_layer"`
	ExpertParamsPerExpert int `json:"expert_params_per_expert"`
}

func NewFFNLayout(cfg Config, exec ExecutionPlan) (FFNLayout, error) {
	if err := cfg.Validate(); err != nil {
		return FFNLayout{}, err
	}
	if len(exec.Steps) == 0 {
		var err error
		exec, err = NewExecutionPlan(cfg)
		if err != nil {
			return FFNLayout{}, err
		}
	}
	denseIntermediate := cfg.IntermediateSize
	if denseIntermediate == 0 {
		denseIntermediate = cfg.HiddenSize * 4
	}
	layout := FFNLayout{
		HiddenSize:        cfg.HiddenSize,
		DenseIntermediate: denseIntermediate,
		MoEIntermediate:   cfg.MoEIntermediateSize,
		DenseLayers:       len(exec.DenseIndices),
		MoELayers:         len(exec.MoEIndices),
		Experts:           cfg.NumExperts,
		ExpertsPerToken:   cfg.NumExpertsPerTok,
	}
	// Gate/up/down projections: hidden->intermediate twice plus intermediate->hidden once.
	layout.DenseParamsPerLayer = 3 * layout.HiddenSize * layout.DenseIntermediate
	layout.ExpertParamsPerExpert = 3 * layout.HiddenSize * layout.MoEIntermediate
	return layout, layout.Validate()
}

func (l FFNLayout) Validate() error {
	if l.HiddenSize <= 0 || l.DenseIntermediate <= 0 || l.MoEIntermediate <= 0 {
		return fmt.Errorf("invalid LFM2 FFN layout dims: %+v", l)
	}
	if l.DenseLayers < 0 || l.MoELayers <= 0 || l.Experts <= 0 || l.ExpertsPerToken <= 0 || l.ExpertsPerToken > l.Experts {
		return fmt.Errorf("invalid LFM2 FFN layout counts: %+v", l)
	}
	if l.DenseParamsPerLayer != 3*l.HiddenSize*l.DenseIntermediate {
		return fmt.Errorf("invalid LFM2 dense params/layer=%d", l.DenseParamsPerLayer)
	}
	if l.ExpertParamsPerExpert != 3*l.HiddenSize*l.MoEIntermediate {
		return fmt.Errorf("invalid LFM2 expert params/expert=%d", l.ExpertParamsPerExpert)
	}
	return nil
}
