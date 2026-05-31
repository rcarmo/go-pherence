package lfm2

import "fmt"

// GenerationExecutionContract is a validation-only contract for the future
// LFM2 CPU/reference generation implementation. It ties token input/output to
// the runtime request plan and context vocabulary bounds.
type GenerationExecutionContract struct {
	Plan         RuntimeRequestPlan `json:"plan"`
	Context      ContextLayout      `json:"context"`
	PromptTokens int                `json:"prompt_tokens"`
	MaxNewTokens int                `json:"max_new_tokens"`
	MaxSequence  int                `json:"max_sequence"`
}

func NewGenerationExecutionContract(plan RuntimeRequestPlan) (GenerationExecutionContract, error) {
	if err := plan.Validate(); err != nil {
		return GenerationExecutionContract{}, err
	}
	contract := GenerationExecutionContract{Plan: plan, Context: plan.Context, PromptTokens: plan.PromptTokens, MaxNewTokens: plan.MaxNewTokens, MaxSequence: plan.MaxSequence}
	return contract, contract.Validate()
}

func (c GenerationExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.Context.Validate(); err != nil {
		return err
	}
	if c.PromptTokens <= 0 || c.MaxNewTokens <= 0 || c.MaxSequence != c.PromptTokens+c.MaxNewTokens || c.MaxSequence != c.Plan.MaxSequence {
		return fmt.Errorf("invalid LFM2 generation contract limits: %+v", c)
	}
	return nil
}

func (c GenerationExecutionContract) ValidatePrompt(tokens []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(tokens) != c.PromptTokens {
		return fmt.Errorf("invalid LFM2 prompt tokens=%d want=%d", len(tokens), c.PromptTokens)
	}
	return c.Context.ValidateSequence(tokens)
}

func (c GenerationExecutionContract) ValidateOutput(tokens []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(tokens) == 0 || len(tokens) > c.MaxNewTokens {
		return fmt.Errorf("invalid LFM2 generated tokens=%d max=%d", len(tokens), c.MaxNewTokens)
	}
	for i, token := range tokens {
		if err := c.Context.ValidateToken(token); err != nil {
			return fmt.Errorf("generated token[%d]: %w", i, err)
		}
	}
	return nil
}
