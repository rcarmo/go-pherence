package lfm2

import "fmt"

// EmbeddingExecutionContract is a validation-only contract for the future LFM2
// CPU/reference embedding stage. It ties prompt tokens to hidden activation
// shape before any convolution/attention/MoE blocks run.
type EmbeddingExecutionContract struct {
	Plan         RuntimeRequestPlan `json:"plan"`
	Context      ContextLayout      `json:"context"`
	Embedding    EmbeddingLayout    `json:"embedding"`
	PromptTokens int                `json:"prompt_tokens"`
	HiddenSize   int                `json:"hidden_size"`
	OutputFloats int                `json:"output_floats"`
}

func NewEmbeddingExecutionContract(cfg Config, plan RuntimeRequestPlan) (EmbeddingExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return EmbeddingExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return EmbeddingExecutionContract{}, err
	}
	embedding, err := NewEmbeddingLayout(cfg)
	if err != nil {
		return EmbeddingExecutionContract{}, err
	}
	contract := EmbeddingExecutionContract{Plan: plan, Context: plan.Context, Embedding: embedding, PromptTokens: plan.PromptTokens, HiddenSize: embedding.HiddenSize, OutputFloats: plan.PromptTokens * embedding.HiddenSize}
	return contract, contract.Validate()
}

func (c EmbeddingExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.Context.Validate(); err != nil {
		return err
	}
	if err := c.Embedding.Validate(); err != nil {
		return err
	}
	if c.PromptTokens <= 0 || c.PromptTokens != c.Plan.PromptTokens || c.HiddenSize != c.Embedding.HiddenSize {
		return fmt.Errorf("invalid LFM2 embedding contract limits: %+v", c)
	}
	if c.OutputFloats != c.PromptTokens*c.HiddenSize {
		return fmt.Errorf("invalid LFM2 embedding output floats=%d want=%d", c.OutputFloats, c.PromptTokens*c.HiddenSize)
	}
	return nil
}

func (c EmbeddingExecutionContract) ValidateInput(tokens []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(tokens) != c.PromptTokens {
		return fmt.Errorf("invalid LFM2 embedding input tokens=%d want=%d", len(tokens), c.PromptTokens)
	}
	return c.Context.ValidateSequence(tokens)
}

func (c EmbeddingExecutionContract) ValidateOutput(hidden []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(hidden) != c.OutputFloats {
		return fmt.Errorf("invalid LFM2 embedding output floats=%d want=%d", len(hidden), c.OutputFloats)
	}
	return nil
}
