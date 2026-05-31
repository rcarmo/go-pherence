package qwen3tts

import "fmt"

// CodePredictorExecutionContract is a validation-only contract for the future
// CPU/reference CodePredictor implementation. It ties semantic-token input to
// bounded acoustic-frame output for the Decoder12Hz handoff.
type CodePredictorExecutionContract struct {
	Plan             RuntimeRequestPlan  `json:"plan"`
	SemanticLayout   SemanticTokenLayout `json:"semantic_layout"`
	FrameLayout      AcousticFrameLayout `json:"frame_layout"`
	MaxFrames        int                 `json:"max_frames"`
	CodesPerFrame    int                 `json:"codes_per_frame"`
	MaxAcousticCodes int                 `json:"max_acoustic_codes"`
}

func NewCodePredictorExecutionContract(cfg ParsedConfig, plan RuntimeRequestPlan) (CodePredictorExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return CodePredictorExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return CodePredictorExecutionContract{}, err
	}
	semantic, err := NewSemanticTokenLayout(cfg)
	if err != nil {
		return CodePredictorExecutionContract{}, err
	}
	frame, err := NewAcousticFrameLayout(cfg)
	if err != nil {
		return CodePredictorExecutionContract{}, err
	}
	contract := CodePredictorExecutionContract{Plan: plan, SemanticLayout: semantic, FrameLayout: frame, MaxFrames: plan.MaxFrames, CodesPerFrame: frame.AcousticCodesPerFrame, MaxAcousticCodes: plan.MaxCodes}
	return contract, contract.Validate()
}

func (c CodePredictorExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.SemanticLayout.Validate(); err != nil {
		return err
	}
	if err := c.FrameLayout.Validate(); err != nil {
		return err
	}
	if c.MaxFrames <= 0 || c.MaxFrames != c.Plan.MaxFrames || c.CodesPerFrame != c.FrameLayout.AcousticCodesPerFrame {
		return fmt.Errorf("invalid Qwen3-TTS CodePredictor contract limits: %+v", c)
	}
	wantCodes := c.MaxFrames * c.CodesPerFrame
	if c.MaxAcousticCodes != wantCodes || c.Plan.MaxCodes != wantCodes {
		return fmt.Errorf("invalid Qwen3-TTS CodePredictor max codes=%d plan=%d want=%d", c.MaxAcousticCodes, c.Plan.MaxCodes, wantCodes)
	}
	return nil
}

func (c CodePredictorExecutionContract) ValidateInput(semantic []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(semantic) == 0 || len(semantic) > c.MaxFrames {
		return fmt.Errorf("invalid Qwen3-TTS CodePredictor semantic tokens=%d max=%d", len(semantic), c.MaxFrames)
	}
	return c.SemanticLayout.ValidateSequence(semantic)
}

func (c CodePredictorExecutionContract) ValidateOutput(acoustic []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(acoustic) == 0 || len(acoustic)%c.CodesPerFrame != 0 || len(acoustic) > c.MaxAcousticCodes {
		return fmt.Errorf("invalid Qwen3-TTS CodePredictor acoustic codes=%d codes_per_frame=%d max=%d", len(acoustic), c.CodesPerFrame, c.MaxAcousticCodes)
	}
	for offset := 0; offset < len(acoustic); offset += c.CodesPerFrame {
		if err := c.FrameLayout.ValidateFrame(acoustic[offset : offset+c.CodesPerFrame]); err != nil {
			return fmt.Errorf("acoustic frame %d: %w", offset/c.CodesPerFrame, err)
		}
	}
	return nil
}
