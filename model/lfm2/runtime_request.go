package lfm2

import "fmt"

// RuntimeRequest captures validation-only inputs and limits for a future LFM2
// generation runtime. It deliberately avoids token generation; it ties prompt
// tokens to context, KV-cache, convolution-state, router, and embedding sizing.
type RuntimeRequest struct {
	Tokens        []uint32 `json:"tokens"`
	MaxNewTokens  int      `json:"max_new_tokens"`
	BytesPerFloat int      `json:"bytes_per_float"`
}

type RuntimeRequestPlan struct {
	Context        ContextLayout `json:"context"`
	PromptTokens   int           `json:"prompt_tokens"`
	MaxNewTokens   int           `json:"max_new_tokens"`
	MaxSequence    int           `json:"max_sequence"`
	BytesPerFloat  int           `json:"bytes_per_float"`
	KVBytes        int64         `json:"kv_bytes"`
	ConvStateBytes int64         `json:"conv_state_bytes"`
	RouterScratch  int           `json:"router_scratch_floats"`
	EmbeddingBytes int64         `json:"embedding_bytes"`
}

func NewRuntimeRequestPlan(cfg Config, req RuntimeRequest) (RuntimeRequestPlan, error) {
	runtimePlan, err := NewRuntimePlan(cfg)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	if len(req.Tokens) == 0 || req.MaxNewTokens <= 0 || req.BytesPerFloat <= 0 {
		return RuntimeRequestPlan{}, fmt.Errorf("invalid LFM2 runtime request limits: tokens=%d max_new_tokens=%d bytes_per_float=%d", len(req.Tokens), req.MaxNewTokens, req.BytesPerFloat)
	}
	if err := runtimePlan.ContextLayout.ValidateSequence(req.Tokens); err != nil {
		return RuntimeRequestPlan{}, err
	}
	maxSeq := len(req.Tokens) + req.MaxNewTokens
	if maxSeq > runtimePlan.ContextLayout.MaxPositionEmbeddings {
		return RuntimeRequestPlan{}, fmt.Errorf("LFM2 request max sequence=%d exceeds context=%d", maxSeq, runtimePlan.ContextLayout.MaxPositionEmbeddings)
	}
	kvBytes, err := runtimePlan.KVBytes(maxSeq, req.BytesPerFloat)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	convBytes, err := runtimePlan.ConvStateBytes(req.BytesPerFloat)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	routerScratch, err := runtimePlan.RouterLayout.ScratchFloats(maxSeq)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	embeddingBytes, err := runtimePlan.EmbeddingLayout.Bytes(req.BytesPerFloat)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	plan := RuntimeRequestPlan{Context: runtimePlan.ContextLayout, PromptTokens: len(req.Tokens), MaxNewTokens: req.MaxNewTokens, MaxSequence: maxSeq, BytesPerFloat: req.BytesPerFloat, KVBytes: kvBytes, ConvStateBytes: convBytes, RouterScratch: routerScratch, EmbeddingBytes: embeddingBytes}
	return plan, plan.Validate()
}

func (p RuntimeRequestPlan) Validate() error {
	if err := p.Context.Validate(); err != nil {
		return err
	}
	if p.PromptTokens <= 0 || p.MaxNewTokens <= 0 || p.MaxSequence != p.PromptTokens+p.MaxNewTokens || p.BytesPerFloat <= 0 {
		return fmt.Errorf("invalid LFM2 runtime request plan limits: %+v", p)
	}
	if p.MaxSequence > p.Context.MaxPositionEmbeddings {
		return fmt.Errorf("LFM2 runtime request max sequence=%d exceeds context=%d", p.MaxSequence, p.Context.MaxPositionEmbeddings)
	}
	if p.KVBytes < 0 || p.ConvStateBytes <= 0 || p.RouterScratch <= 0 || p.EmbeddingBytes <= 0 {
		return fmt.Errorf("invalid LFM2 runtime request sizing: %+v", p)
	}
	return nil
}
