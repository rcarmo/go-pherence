package model

import "fmt"

type MTPGraphGenerationOptions struct {
	MaxTokens int
	Policy    MTPAdaptiveDraftPolicy
	Stats     MTPSpeculationStats
}

type MTPGraphGenerationResult struct {
	Output                     []int
	Stats                      MTPSpeculationStats
	FinalState                 MTPDrafterState
	Steps                      []MTPKVCommitPlan
	Capabilities               MTPGraphCapabilities
	MissingForPublicGeneration []string
}

// NewCPUDecodeStateFromMTPPromptContext creates a graph-decode state from a
// prompt context produced by BuildMTPPromptContext. It copies tokens and KV so a
// failed speculative step can restore safely without mutating the prompt context.
func NewCPUDecodeStateFromMTPPromptContext(m *LlamaModel, ctx MTPPromptContext, maxTokens int) (*CPUDecodeState, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens=%d must be >= 0", maxTokens)
	}
	if len(ctx.Tokens) == 0 || ctx.SeqLen != len(ctx.Tokens) {
		return nil, fmt.Errorf("invalid MTP prompt context tokens=%d seqLen=%d", len(ctx.Tokens), ctx.SeqLen)
	}
	if ctx.PreviousToken != ctx.Tokens[len(ctx.Tokens)-1] {
		return nil, fmt.Errorf("prompt context previous token=%d, want final token %d", ctx.PreviousToken, ctx.Tokens[len(ctx.Tokens)-1])
	}
	if m.Config.HiddenSize <= 0 || len(ctx.Activation) != m.Config.HiddenSize {
		return nil, fmt.Errorf("prompt context activation len=%d, want hidden=%d", len(ctx.Activation), m.Config.HiddenSize)
	}
	state, err := NewCPUDecodeStateForSpeculative(m, ctx.Tokens, maxTokens, "mtp-graph")
	if err != nil {
		return nil, err
	}
	if len(ctx.KVCacheK) != len(state.KVCacheK) || len(ctx.KVCacheV) != len(state.KVCacheV) {
		return nil, fmt.Errorf("prompt context KV layers K/V=%d/%d, want %d", len(ctx.KVCacheK), len(ctx.KVCacheV), len(state.KVCacheK))
	}
	for l, dim := range state.KVDims {
		if dim <= 0 {
			if len(ctx.KVCacheK[l]) != 0 || len(ctx.KVCacheV[l]) != 0 {
				return nil, fmt.Errorf("prompt context shared/non-KV layer %d has K/V entries %d/%d", l, len(ctx.KVCacheK[l]), len(ctx.KVCacheV[l]))
			}
			continue
		}
		want := ctx.SeqLen * dim
		if len(ctx.KVCacheK[l]) != want || len(ctx.KVCacheV[l]) != want {
			return nil, fmt.Errorf("prompt context layer %d KV K/V=%d/%d, want %d", l, len(ctx.KVCacheK[l]), len(ctx.KVCacheV[l]), want)
		}
		state.KVCacheK[l] = append(state.KVCacheK[l], ctx.KVCacheK[l]...)
		state.KVCacheV[l] = append(state.KVCacheV[l], ctx.KVCacheV[l]...)
	}
	return state, nil
}

// GenerateMTPGraphFromPromptContext runs the internal graph-backed MTP cycle
// from a prefilled prompt context. It is the model-level production contract for
// future CLI/API wiring; current real Gemma4 verifier limitations still apply.
func (m *LlamaModel) GenerateMTPGraphFromPromptContext(d *Gemma4MTPDrafter, ctx MTPPromptContext, externalKV *MTPDrafterExternalKV, opts MTPGraphGenerationOptions) (MTPGraphGenerationResult, error) {
	if m == nil {
		return MTPGraphGenerationResult{}, fmt.Errorf("nil model")
	}
	if d == nil {
		return MTPGraphGenerationResult{}, fmt.Errorf("nil drafter")
	}
	if opts.MaxTokens < 0 {
		return MTPGraphGenerationResult{}, fmt.Errorf("maxTokens=%d must be >= 0", opts.MaxTokens)
	}
	decode, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, opts.MaxTokens)
	if err != nil {
		return MTPGraphGenerationResult{}, err
	}
	state, err := NewMTPDrafterState(ctx.PreviousToken, ctx.Activation, d.BackboneHiddenSize)
	if err != nil {
		return MTPGraphGenerationResult{}, err
	}
	stats := opts.Stats
	commits := make([]MTPKVCommitPlan, 0)
	for len(decode.Output)-len(ctx.Tokens) < opts.MaxTokens {
		remaining := opts.MaxTokens - (len(decode.Output) - len(ctx.Tokens))
		if remaining <= 1 {
			// A graph verifier pass with G drafts can emit up to G+1 tokens, so a
			// one-token tail is ordinary greedy decode territory. This mirrors
			// speculative generation fallback behavior while preserving the graph
			// contract for multi-token steps.
			if _, err := decode.DecodeOneGreedy(); err != nil {
				return MTPGraphGenerationResult{}, err
			}
			break
		}
		stepExternalKV, err := mtpExternalKVForDecodeState(decode, externalKV)
		if err != nil {
			return MTPGraphGenerationResult{}, err
		}
		step, err := decode.RunMTPGraphDecodeStep(d, state, stepExternalKV, MTPGraphDecodeStepOptions{RemainingTokens: remaining, Policy: opts.Policy}, stats)
		if err != nil {
			return MTPGraphGenerationResult{}, err
		}
		state = step.FinalState
		stats = step.Stats
		commits = append(commits, step.Commit)
		if len(step.Commit.OutputTokens) == 0 {
			break
		}
	}
	caps := Gemma4MTPGraphCapabilities()
	return MTPGraphGenerationResult{Output: append([]int(nil), decode.Output...), Stats: stats, FinalState: state, Steps: commits, Capabilities: caps, MissingForPublicGeneration: caps.MissingForPublicGeneration()}, nil
}

func mtpExternalKVForDecodeState(decode *CPUDecodeState, base *MTPDrafterExternalKV) (*MTPDrafterExternalKV, error) {
	if base == nil {
		return nil, nil
	}
	if decode == nil {
		return nil, fmt.Errorf("nil decode state")
	}
	if len(decode.KVCacheK) != len(base.K) || len(decode.KVCacheV) != len(base.V) {
		return nil, fmt.Errorf("decode KV layers K/V=%d/%d, base K/V=%d/%d", len(decode.KVCacheK), len(decode.KVCacheV), len(base.K), len(base.V))
	}
	return &MTPDrafterExternalKV{
		K:            decode.KVCacheK,
		V:            decode.KVCacheV,
		SourceLayers: append([]int(nil), base.SourceLayers...),
		SeqLen:       len(decode.Output),
	}, nil
}
