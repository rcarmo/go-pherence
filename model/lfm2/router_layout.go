package lfm2

import "fmt"

// RouterLayout captures the MoE router projection and logits contract for LFM2
// routed layers. It complements RoutingPlan by sizing the actual hidden→expert
// router matrix and per-token top-k scratch before expert runtime dispatch.
type RouterLayout struct {
	HiddenSize         int  `json:"hidden_size"`
	Experts            int  `json:"experts"`
	ExpertsPerToken    int  `json:"experts_per_token"`
	MoELayers          int  `json:"moe_layers"`
	UseExpertBias      bool `json:"use_expert_bias"`
	RouterWeightFloats int  `json:"router_weight_floats"`
	RouterBiasFloats   int  `json:"router_bias_floats"`
	FloatsPerLayer     int  `json:"floats_per_layer"`
	TotalRouterFloats  int  `json:"total_router_floats"`
	LogitsPerToken     int  `json:"logits_per_token"`
	TopKPerToken       int  `json:"topk_per_token"`
}

func NewRouterLayout(cfg Config, exec ExecutionPlan) (RouterLayout, error) {
	if err := cfg.Validate(); err != nil {
		return RouterLayout{}, err
	}
	if len(exec.Steps) == 0 {
		var err error
		exec, err = NewExecutionPlan(cfg)
		if err != nil {
			return RouterLayout{}, err
		}
	}
	if err := exec.Validate(cfg.NumHiddenLayers); err != nil {
		return RouterLayout{}, err
	}
	layout := RouterLayout{
		HiddenSize:         cfg.HiddenSize,
		Experts:            cfg.NumExperts,
		ExpertsPerToken:    cfg.NumExpertsPerTok,
		MoELayers:          len(exec.MoEIndices),
		UseExpertBias:      cfg.UseExpertBias,
		RouterWeightFloats: cfg.NumExperts * cfg.HiddenSize,
		LogitsPerToken:     cfg.NumExperts,
		TopKPerToken:       cfg.NumExpertsPerTok,
	}
	if cfg.UseExpertBias {
		layout.RouterBiasFloats = cfg.NumExperts
	}
	layout.FloatsPerLayer = layout.RouterWeightFloats + layout.RouterBiasFloats
	layout.TotalRouterFloats = layout.FloatsPerLayer * layout.MoELayers
	return layout, layout.Validate()
}

func (l RouterLayout) Validate() error {
	if l.HiddenSize <= 0 || l.Experts <= 0 || l.ExpertsPerToken <= 0 || l.ExpertsPerToken > l.Experts || l.MoELayers < 0 {
		return fmt.Errorf("invalid LFM2 router layout dims: %+v", l)
	}
	wantWeight := l.Experts * l.HiddenSize
	if l.RouterWeightFloats != wantWeight {
		return fmt.Errorf("invalid LFM2 router weight floats=%d want=%d", l.RouterWeightFloats, wantWeight)
	}
	wantBias := 0
	if l.UseExpertBias {
		wantBias = l.Experts
	}
	if l.RouterBiasFloats != wantBias {
		return fmt.Errorf("invalid LFM2 router bias floats=%d want=%d", l.RouterBiasFloats, wantBias)
	}
	wantLayer := l.RouterWeightFloats + l.RouterBiasFloats
	if l.FloatsPerLayer != wantLayer {
		return fmt.Errorf("invalid LFM2 router floats/layer=%d want=%d", l.FloatsPerLayer, wantLayer)
	}
	if l.TotalRouterFloats != l.FloatsPerLayer*l.MoELayers {
		return fmt.Errorf("invalid LFM2 router total floats=%d want=%d", l.TotalRouterFloats, l.FloatsPerLayer*l.MoELayers)
	}
	if l.LogitsPerToken != l.Experts || l.TopKPerToken != l.ExpertsPerToken {
		return fmt.Errorf("invalid LFM2 router token outputs: %+v", l)
	}
	return nil
}

func (l RouterLayout) ScratchFloats(tokens int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if tokens < 0 {
		return 0, fmt.Errorf("invalid LFM2 router token count=%d", tokens)
	}
	return tokens * (l.LogitsPerToken + 2*l.TopKPerToken), nil
}
