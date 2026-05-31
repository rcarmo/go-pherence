package lfm2

import "fmt"

// MoEExecutionContract is a validation-only contract for the future LFM2
// CPU/reference router/top-k/expert FFN stage. It ties hidden activations,
// router scratch, and active expert outputs to exact per-token dimensions.
type MoEExecutionContract struct {
	Plan           RuntimeRequestPlan `json:"plan"`
	RouterLayout   RouterLayout       `json:"router_layout"`
	FFNLayout      FFNLayout          `json:"ffn_layout"`
	SequenceTokens int                `json:"sequence_tokens"`
	HiddenSize     int                `json:"hidden_size"`
	HiddenFloats   int                `json:"hidden_floats"`
	RouterScratch  int                `json:"router_scratch_floats"`
	TopKPerToken   int                `json:"topk_per_token"`
}

func NewMoEExecutionContract(cfg Config, plan RuntimeRequestPlan) (MoEExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return MoEExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return MoEExecutionContract{}, err
	}
	runtimePlan, err := NewRuntimePlan(cfg)
	if err != nil {
		return MoEExecutionContract{}, err
	}
	contract := MoEExecutionContract{Plan: plan, RouterLayout: runtimePlan.RouterLayout, FFNLayout: runtimePlan.FFNLayout, SequenceTokens: plan.MaxSequence, HiddenSize: runtimePlan.HiddenSize, HiddenFloats: plan.MaxSequence * runtimePlan.HiddenSize, RouterScratch: plan.RouterScratch, TopKPerToken: runtimePlan.RouterLayout.TopKPerToken}
	return contract, contract.Validate()
}

func (c MoEExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.RouterLayout.Validate(); err != nil {
		return err
	}
	if err := c.FFNLayout.Validate(); err != nil {
		return err
	}
	if c.SequenceTokens <= 0 || c.SequenceTokens != c.Plan.MaxSequence || c.HiddenSize <= 0 || c.HiddenSize != c.RouterLayout.HiddenSize || c.HiddenSize != c.FFNLayout.HiddenSize {
		return fmt.Errorf("invalid LFM2 MoE contract dims: %+v", c)
	}
	if c.HiddenFloats != c.SequenceTokens*c.HiddenSize {
		return fmt.Errorf("invalid LFM2 MoE hidden floats=%d want=%d", c.HiddenFloats, c.SequenceTokens*c.HiddenSize)
	}
	wantScratch, err := c.RouterLayout.ScratchFloats(c.SequenceTokens)
	if err != nil {
		return err
	}
	if c.RouterScratch != wantScratch || c.Plan.RouterScratch != wantScratch {
		return fmt.Errorf("invalid LFM2 MoE router scratch=%d plan=%d want=%d", c.RouterScratch, c.Plan.RouterScratch, wantScratch)
	}
	if c.TopKPerToken != c.RouterLayout.TopKPerToken || c.TopKPerToken != c.FFNLayout.ExpertsPerToken {
		return fmt.Errorf("invalid LFM2 MoE top-k=%d router=%d ffn=%d", c.TopKPerToken, c.RouterLayout.TopKPerToken, c.FFNLayout.ExpertsPerToken)
	}
	return nil
}

func (c MoEExecutionContract) ValidateInput(hidden []float32, routerScratch []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(hidden) != c.HiddenFloats {
		return fmt.Errorf("invalid LFM2 MoE hidden input floats=%d want=%d", len(hidden), c.HiddenFloats)
	}
	if len(routerScratch) != c.RouterScratch {
		return fmt.Errorf("invalid LFM2 MoE router scratch floats=%d want=%d", len(routerScratch), c.RouterScratch)
	}
	return nil
}

func (c MoEExecutionContract) ValidateOutput(hidden []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(hidden) != c.HiddenFloats {
		return fmt.Errorf("invalid LFM2 MoE hidden output floats=%d want=%d", len(hidden), c.HiddenFloats)
	}
	return nil
}
