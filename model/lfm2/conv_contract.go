package lfm2

import "fmt"

// ConvExecutionContract is a validation-only contract for the future LFM2
// CPU/reference short-convolution stage. It ties hidden activations, conv state,
// and conv projection layouts to exact float counts.
type ConvExecutionContract struct {
	Plan             RuntimeRequestPlan   `json:"plan"`
	StateLayout      ConvStateLayout      `json:"state_layout"`
	ProjectionLayout ConvProjectionLayout `json:"projection_layout"`
	SequenceTokens   int                  `json:"sequence_tokens"`
	HiddenSize       int                  `json:"hidden_size"`
	HiddenFloats     int                  `json:"hidden_floats"`
	StateFloats      int                  `json:"state_floats"`
}

func NewConvExecutionContract(cfg Config, plan RuntimeRequestPlan) (ConvExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return ConvExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return ConvExecutionContract{}, err
	}
	runtimePlan, err := NewRuntimePlan(cfg)
	if err != nil {
		return ConvExecutionContract{}, err
	}
	contract := ConvExecutionContract{Plan: plan, StateLayout: runtimePlan.ConvStateLayout, ProjectionLayout: runtimePlan.ConvProjLayout, SequenceTokens: plan.MaxSequence, HiddenSize: runtimePlan.HiddenSize, HiddenFloats: plan.MaxSequence * runtimePlan.HiddenSize, StateFloats: runtimePlan.ConvStateFloats}
	return contract, contract.Validate()
}

func (c ConvExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.StateLayout.Validate(); err != nil {
		return err
	}
	if err := c.ProjectionLayout.Validate(); err != nil {
		return err
	}
	if c.SequenceTokens <= 0 || c.SequenceTokens != c.Plan.MaxSequence || c.HiddenSize <= 0 || c.HiddenSize != c.StateLayout.HiddenSize || c.HiddenSize != c.ProjectionLayout.HiddenSize {
		return fmt.Errorf("invalid LFM2 conv contract dims: %+v", c)
	}
	if c.HiddenFloats != c.SequenceTokens*c.HiddenSize {
		return fmt.Errorf("invalid LFM2 conv hidden floats=%d want=%d", c.HiddenFloats, c.SequenceTokens*c.HiddenSize)
	}
	if c.StateFloats != c.StateLayout.TotalFloats {
		return fmt.Errorf("invalid LFM2 conv state floats=%d want=%d", c.StateFloats, c.StateLayout.TotalFloats)
	}
	return nil
}

func (c ConvExecutionContract) ValidateInput(hidden []float32, state []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(hidden) != c.HiddenFloats {
		return fmt.Errorf("invalid LFM2 conv hidden input floats=%d want=%d", len(hidden), c.HiddenFloats)
	}
	if len(state) != c.StateFloats {
		return fmt.Errorf("invalid LFM2 conv state input floats=%d want=%d", len(state), c.StateFloats)
	}
	return nil
}

func (c ConvExecutionContract) ValidateOutput(hidden []float32, state []float32) error {
	if err := c.ValidateInput(hidden, state); err != nil {
		return err
	}
	return nil
}
