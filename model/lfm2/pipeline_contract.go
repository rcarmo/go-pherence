package lfm2

import "errors"

var ErrPipelineContractMismatch = errors.New("lfm2 pipeline execution contract mismatch")

// PipelineExecutionContract groups the validation-only stage contracts for the
// future LFM2 CPU/reference generation pipeline. It ensures embedding, conv,
// attention, MoE, and final token-generation contracts agree on one request.
type PipelineExecutionContract struct {
	Generation GenerationExecutionContract `json:"generation"`
	Embedding  EmbeddingExecutionContract  `json:"embedding"`
	Conv       ConvExecutionContract       `json:"conv"`
	Attention  AttentionExecutionContract  `json:"attention"`
	MoE        MoEExecutionContract        `json:"moe"`
}

func NewPipelineExecutionContract(cfg Config, plan RuntimeRequestPlan) (PipelineExecutionContract, error) {
	generation, err := NewGenerationExecutionContract(plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	embedding, err := NewEmbeddingExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	conv, err := NewConvExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	attention, err := NewAttentionExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	moe, err := NewMoEExecutionContract(cfg, plan)
	if err != nil {
		return PipelineExecutionContract{}, err
	}
	contract := PipelineExecutionContract{Generation: generation, Embedding: embedding, Conv: conv, Attention: attention, MoE: moe}
	return contract, contract.Validate()
}

func (c PipelineExecutionContract) Validate() error {
	if err := c.Generation.Validate(); err != nil {
		return err
	}
	if err := c.Embedding.Validate(); err != nil {
		return err
	}
	if err := c.Conv.Validate(); err != nil {
		return err
	}
	if err := c.Attention.Validate(); err != nil {
		return err
	}
	if err := c.MoE.Validate(); err != nil {
		return err
	}
	// Cross-stage shape consistency.
	if c.Embedding.HiddenSize != c.Conv.HiddenSize || c.Conv.HiddenSize != c.Attention.HiddenSize || c.Attention.HiddenSize != c.MoE.HiddenSize {
		return ErrPipelineContractMismatch
	}
	if c.Conv.HiddenFloats != c.Attention.HiddenFloats || c.Attention.HiddenFloats != c.MoE.HiddenFloats {
		return ErrPipelineContractMismatch
	}
	if c.Generation.MaxSequence != c.Conv.SequenceTokens || c.Conv.SequenceTokens != c.Attention.SequenceTokens || c.Attention.SequenceTokens != c.MoE.SequenceTokens {
		return ErrPipelineContractMismatch
	}
	return nil
}
