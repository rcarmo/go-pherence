package lfm2

import "fmt"

// AttentionExecutionContract is a validation-only contract for the future LFM2
// CPU/reference full-attention stage. It ties hidden activations to the
// full-attention KV-cache layout and projection sizing.
type AttentionExecutionContract struct {
	Plan             RuntimeRequestPlan        `json:"plan"`
	KVLayout         AttentionKVLayout         `json:"kv_layout"`
	ProjectionLayout AttentionProjectionLayout `json:"projection_layout"`
	SequenceTokens   int                       `json:"sequence_tokens"`
	HiddenSize       int                       `json:"hidden_size"`
	HiddenFloats     int                       `json:"hidden_floats"`
	KVFloats         int                       `json:"kv_floats"`
}

func NewAttentionExecutionContract(cfg Config, plan RuntimeRequestPlan) (AttentionExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return AttentionExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return AttentionExecutionContract{}, err
	}
	runtimePlan, err := NewRuntimePlan(cfg)
	if err != nil {
		return AttentionExecutionContract{}, err
	}
	contract := AttentionExecutionContract{Plan: plan, KVLayout: runtimePlan.AttentionKVLayout, ProjectionLayout: runtimePlan.AttentionProjLayout, SequenceTokens: plan.MaxSequence, HiddenSize: runtimePlan.HiddenSize, HiddenFloats: plan.MaxSequence * runtimePlan.HiddenSize, KVFloats: plan.MaxSequence * runtimePlan.AttentionKVLayout.FloatsPerToken}
	return contract, contract.Validate()
}

func (c AttentionExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.KVLayout.Validate(); err != nil {
		return err
	}
	if err := c.ProjectionLayout.Validate(); err != nil {
		return err
	}
	if c.SequenceTokens <= 0 || c.SequenceTokens != c.Plan.MaxSequence || c.HiddenSize <= 0 || c.HiddenSize != c.ProjectionLayout.HiddenSize {
		return fmt.Errorf("invalid LFM2 attention contract dims: %+v", c)
	}
	if c.HiddenFloats != c.SequenceTokens*c.HiddenSize {
		return fmt.Errorf("invalid LFM2 attention hidden floats=%d want=%d", c.HiddenFloats, c.SequenceTokens*c.HiddenSize)
	}
	wantKV := c.SequenceTokens * c.KVLayout.FloatsPerToken
	if c.KVFloats != wantKV {
		return fmt.Errorf("invalid LFM2 attention KV floats=%d want=%d", c.KVFloats, wantKV)
	}
	return nil
}

func (c AttentionExecutionContract) ValidateInput(hidden []float32, kv []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(hidden) != c.HiddenFloats {
		return fmt.Errorf("invalid LFM2 attention hidden input floats=%d want=%d", len(hidden), c.HiddenFloats)
	}
	if len(kv) != c.KVFloats {
		return fmt.Errorf("invalid LFM2 attention KV input floats=%d want=%d", len(kv), c.KVFloats)
	}
	return nil
}

func (c AttentionExecutionContract) ValidateOutput(hidden []float32, kv []float32) error {
	if err := c.ValidateInput(hidden, kv); err != nil {
		return err
	}
	return nil
}
