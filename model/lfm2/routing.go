package lfm2

import "fmt"

// RoutingPlan captures the MoE router contract without executing it. Runtime
// code should use this as the source of truth for top-k selection semantics and
// routed layer counts.
type RoutingPlan struct {
	Experts             int     `json:"experts"`
	ExpertsPerToken     int     `json:"experts_per_token"`
	MoELayers           int     `json:"moe_layers"`
	DenseLayers         int     `json:"dense_layers"`
	MoEIntermediate     int     `json:"moe_intermediate"`
	NormalizeTopK       bool    `json:"normalize_topk"`
	UseExpertBias       bool    `json:"use_expert_bias"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor"`
}

func NewRoutingPlan(cfg Config, exec ExecutionPlan) (RoutingPlan, error) {
	if err := cfg.Validate(); err != nil {
		return RoutingPlan{}, err
	}
	if len(exec.Steps) == 0 {
		var err error
		exec, err = NewExecutionPlan(cfg)
		if err != nil {
			return RoutingPlan{}, err
		}
	}
	plan := RoutingPlan{
		Experts:             cfg.NumExperts,
		ExpertsPerToken:     cfg.NumExpertsPerTok,
		MoELayers:           len(exec.MoEIndices),
		DenseLayers:         len(exec.DenseIndices),
		MoEIntermediate:     cfg.MoEIntermediateSize,
		NormalizeTopK:       cfg.NormTopKProb,
		UseExpertBias:       cfg.UseExpertBias,
		RoutedScalingFactor: cfg.RoutedScalingFactor,
	}
	return plan, plan.Validate()
}

func (p RoutingPlan) Validate() error {
	if p.Experts <= 0 || p.ExpertsPerToken <= 0 || p.ExpertsPerToken > p.Experts {
		return fmt.Errorf("invalid LFM2 routing experts=%d active=%d", p.Experts, p.ExpertsPerToken)
	}
	if p.MoELayers <= 0 || p.DenseLayers < 0 || p.MoEIntermediate <= 0 {
		return fmt.Errorf("invalid LFM2 routing layer/intermediate plan: %+v", p)
	}
	if p.RoutedScalingFactor < 0 {
		return fmt.Errorf("invalid LFM2 routed scaling factor=%g", p.RoutedScalingFactor)
	}
	return nil
}
