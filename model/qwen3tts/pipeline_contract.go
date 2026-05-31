package qwen3tts

import "errors"

var ErrPipelineContractMismatch = errors.New("qwen3tts pipeline execution contract mismatch")

// PipelineExecutionContract groups the validation-only stage contracts for the
// future Qwen3-TTS CPU/reference synthesis pipeline. It ensures Talker,
// CodePredictor, and Decoder12Hz contracts agree on one request.
type PipelineExecutionContract struct {
	Plan          RuntimeRequestPlan             `json:"plan"`
	Talker        TalkerExecutionContract        `json:"talker"`
	CodePredictor CodePredictorExecutionContract `json:"code_predictor"`
	Decoder12Hz   Decoder12HzExecutionContract   `json:"decoder12hz"`
}

func NewPipelineExecutionContract(cfg ParsedConfig, plan RuntimeRequestPlan) (PipelineExecutionContract, error) {
	if err := cfg.Validate(); err != nil {
		return PipelineExecutionContract{}, err
	}
	if err := plan.Validate(); err != nil {
		return PipelineExecutionContract{}, err
	}
	talker, err := NewTalkerExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	codePredictor, err := NewCodePredictorExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	decoder, err := NewDecoder12HzExecutionContract(plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	contract := PipelineExecutionContract{Plan: plan, Talker: talker, CodePredictor: codePredictor, Decoder12Hz: decoder}
	return contract, contract.Validate()
}

func (c PipelineExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.Talker.Validate(); err != nil {
		return err
	}
	if err := c.CodePredictor.Validate(); err != nil {
		return err
	}
	if err := c.Decoder12Hz.Validate(); err != nil {
		return err
	}
	if c.Talker.MaxTokens != c.Plan.MaxFrames || c.CodePredictor.MaxFrames != c.Plan.MaxFrames || c.Decoder12Hz.MaxFrames != c.Plan.MaxFrames {
		return ErrPipelineContractMismatch
	}
	if c.CodePredictor.CodesPerFrame != c.Decoder12Hz.CodesPerFrame || c.CodePredictor.MaxAcousticCodes != c.Decoder12Hz.MaxAcousticCodes {
		return ErrPipelineContractMismatch
	}
	if c.Decoder12Hz.MaxSamples != c.Plan.MaxSamples {
		return ErrPipelineContractMismatch
	}
	return nil
}

func (c PipelineExecutionContract) ValidateStageOutputs(semantic []uint32, acoustic []uint32, samples []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if err := c.Talker.ValidateOutput(semantic); err != nil {
		return err
	}
	if err := c.CodePredictor.ValidateInput(semantic); err != nil {
		return err
	}
	if err := c.CodePredictor.ValidateOutput(acoustic); err != nil {
		return err
	}
	if err := c.Decoder12Hz.ValidateInput(acoustic); err != nil {
		return err
	}
	return c.Decoder12Hz.ValidateOutput(samples)
}
