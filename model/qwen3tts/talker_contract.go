package qwen3tts

import "fmt"

// TalkerExecutionContract is a validation-only contract for the future
// CPU/reference Talker implementation. It ties a planned synthesis request to
// the semantic token layout and bounded output sequence the Talker must return.
type TalkerExecutionContract struct {
	Plan           RuntimeRequestPlan  `json:"plan"`
	SemanticLayout SemanticTokenLayout `json:"semantic_layout"`
	MaxTokens      int                 `json:"max_tokens"`
}

func NewTalkerExecutionContract(cfg ParsedConfig, plan RuntimeRequestPlan) (TalkerExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return TalkerExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return TalkerExecutionContract{}, err
	}
	semantic, err := NewSemanticTokenLayout(cfg)
	if err != nil {
		return TalkerExecutionContract{}, err
	}
	contract := TalkerExecutionContract{Plan: plan, SemanticLayout: semantic, MaxTokens: plan.MaxFrames}
	return contract, contract.Validate()
}

func (c TalkerExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.SemanticLayout.Validate(); err != nil {
		return err
	}
	if c.MaxTokens <= 0 || c.MaxTokens > c.Plan.MaxFrames {
		return fmt.Errorf("invalid Qwen3-TTS Talker max tokens=%d request_frames=%d", c.MaxTokens, c.Plan.MaxFrames)
	}
	return nil
}

func (c TalkerExecutionContract) ValidateOutput(tokens []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(tokens) == 0 || len(tokens) > c.MaxTokens {
		return fmt.Errorf("invalid Qwen3-TTS Talker output tokens=%d max=%d", len(tokens), c.MaxTokens)
	}
	return c.SemanticLayout.ValidateSequence(tokens)
}
